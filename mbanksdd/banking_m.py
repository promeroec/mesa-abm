from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np


class Depositor:
    """
    Simple depositor object (not a Mesa Agent).
    type: 'patient' or 'impatient'
    """
    def __init__(self, unique_id, bank_id, depositor_type, initial_deposit=1.0):
        self.id = unique_id
        self.bank_id = bank_id
        self.type = depositor_type
        self.balance = float(initial_deposit)
        self.has_withdrawn = False

    def wants_to_withdraw(self, bank, model, position_in_queue):
        """
        Withdrawal decision rule.

        - Impatient depositors always try to withdraw when they reach the teller.
        - Patient depositors may run if, among those *before* them in the queue,
          the fraction that has already withdrawn is above a panic threshold.
        """
        if self.has_withdrawn or self.balance <= 0 or bank.failed:
            return False

        # Impatient always withdraw
        if self.type == "impatient":
            return True

        # Patient: look at previous customers in this bank's queue
        num_before = position_in_queue
        if num_before <= 0:
            return False

        withdrawn_before = sum(
            1 for dep_id in bank.customers[:position_in_queue]
            if model.depositors[dep_id].has_withdrawn
        )
        frac_withdrawn = withdrawn_before / float(num_before)
        return frac_withdrawn >= model.panic_threshold


class BankAgent(Agent):
    """
    Bank agent implementing a Diamond–Dybvig style deposit contract with reserves
    and an interbank market for liquidity (no central bank).

    We also track interbank assets and liabilities so that a bank's failure can
    propagate losses to creditor banks.
    """
    def __init__(self, unique_id, model, initial_deposits, reserve_ratio):
        super().__init__(unique_id, model)
        self.initial_deposits = float(initial_deposits)
        self.reserve_ratio = float(reserve_ratio)
        self.reserves = self.reserve_ratio * self.initial_deposits
        self.illiquid = (1.0 - self.reserve_ratio) * self.initial_deposits
        self.failed = False

        # customers is a list of depositor IDs
        self.customers = []
        self.queue_index = 0  # how many customers have been seen so far

        # Interbank positions (from this bank's perspective)
        # assets: loans to other banks
        # liabilities: borrowing from other banks
        self.interbank_assets = {}      # { counterparty_id: amount }
        self.interbank_liabilities = {} # { counterparty_id: amount }

    def add_customer(self, depositor_id):
        self.customers.append(depositor_id)

    @property
    def bank_id(self):
        return self.unique_id

    def step(self):
        """
        In each period, each bank serves at most one customer in its queue.
        """
        if self.failed:
            return

        if self.queue_index >= len(self.customers):
            return  # no more customers to serve

        position_in_queue = self.queue_index
        dep_id = self.customers[position_in_queue]
        depositor = self.model.depositors[dep_id]

        if depositor.wants_to_withdraw(self, self.model, position_in_queue):
            self.process_withdrawal(depositor)

        # After the attempt (whether successful or not), move queue forward
        self.queue_index += 1

    def process_withdrawal(self, depositor):
        """
        Try to pay out the depositor. If reserves are insufficient,
        try to borrow in the interbank market. If still insufficient,
        the bank fails.
        """
        amount = depositor.balance
        if amount <= 0:
            return

        # Directly from reserves
        if self.reserves >= amount:
            self.reserves -= amount
            depositor.balance = 0.0
            depositor.has_withdrawn = True
            return

        # Need to borrow from interbank market?
        shortage = amount - self.reserves
        borrowed = 0.0
        if self.model.has_interbank:
            borrowed = self.model.borrow_from_interbank(self, shortage)

        total_liquidity = self.reserves + borrowed
        if total_liquidity >= amount:
            # Bank survives this withdrawal
            self.reserves = total_liquidity - amount
            depositor.balance = 0.0
            depositor.has_withdrawn = True
        else:
            # Bank fails: cannot honor the current withdrawal
            self.failed = True
            self.model.register_bank_failure(self)


class MultiBankModel(Model):
    """
    Multi-bank Diamond–Dybvig style ABM based on Romero (2009), without a central bank.

    Arrangements (industrial organization only; same parameters otherwise):

      - "no_interbank":
          4 banks, 40 depositors
          10 depositors per bank, no interbank links.

      - "symmetric":
          4 banks, 40 depositors
          10 depositors per bank, fully connected interbank market.

      - "big_bank":
          4 banks, 40 depositors
          One "big" bank with 20 customers and three small banks with 10, 5, 5
          customers, star-shaped interbank network centered on the big bank.

    All cases share:
      * total_depositors = 40
      * same reserve_ratio, panic_threshold, etc.

    The goal is to reproduce the *qualitative* ordering in frequencies of runs:
      isolated banks > big-bank market > symmetric interbank market,
    and in the big-bank case runs should typically involve the big bank and
    spill over to some small banks through interbank exposures.
    """

    def __init__(
        self,
        arrangement="no_interbank",
        num_banks=4,
        total_depositors=40,
        num_patient=16,
        num_impatient=24,
        reserve_ratio=0.5,
        deposit_per_depositor=1.0,
        panic_threshold=0.5,
        max_steps=25,
        seed=None,
    ):
        super().__init__()
        if seed is not None:
            self.seed = seed
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

        self.arrangement = arrangement
        self.num_banks = num_banks
        self.total_depositors = total_depositors
        self.deposit_per_depositor = deposit_per_depositor
        self.reserve_ratio = reserve_ratio
        self.panic_threshold = panic_threshold
        self.max_steps = max_steps

        # We use the SAME composition (16 patient, 24 impatient) in all models
        # so that only industrial organization differs.
        if num_patient + num_impatient != total_depositors:
            raise ValueError("num_patient + num_impatient must equal total_depositors.")
        self.num_patient = num_patient
        self.num_impatient = num_impatient

        if self.total_depositors != 40:
            raise ValueError("For this Romero-style setup, use total_depositors = 40.")

        # Flags and trackers
        self.has_interbank = arrangement in ("symmetric", "big_bank")
        self.current_step = 0
        self.bank_failures = []
        self.first_failure_step = None

        # Mesa scheduler
        self.schedule = BaseScheduler(self)

        # Containers
        self.banks = {}
        self.depositors = {}

        # Create environment
        self._create_banks()
        self._create_depositors_and_assign()
        self._create_interbank_network()

        # Data collector: easy hooks for plots/tables and Monte Carlo
        self.datacollector = DataCollector(
            model_reporters={
                "num_failed_banks": lambda m: len(m.bank_failures),
                "first_failure_step": lambda m: m.first_failure_step
                if m.first_failure_step is not None else -1,
                "num_patient": lambda m: m.num_patient,
                "num_impatient": lambda m: m.num_impatient,
            }
        )

    # ---------- Environment construction helpers ----------

    def _create_banks(self):
        """
        Create banks with placeholder initial deposits; actual deposits/reserves
        are set after we assign customers.
        """
        for bank_id in range(self.num_banks):
            bank = BankAgent(bank_id, self, initial_deposits=0.0,
                             reserve_ratio=self.reserve_ratio)
            self.banks[bank_id] = bank
            self.schedule.add(bank)

    def _create_depositors_and_assign(self):
        """
        Create depositors and assign them to banks according to arrangement.
        After assignment, update each bank's deposit base and reserves.
        """
        # Build list of depositor types (same across arrangements)
        types = ["patient"] * self.num_patient + ["impatient"] * self.num_impatient
        self._rng.shuffle(types)

        # Customers per bank by industrial organization
        if self.arrangement in ("no_interbank", "symmetric"):
            # 4 × 10 = 40
            customers_per_bank = [10, 10, 10, 10]
        elif self.arrangement == "big_bank":
            # 20 + 10 + 5 + 5 = 40
            # bank 0 is the "big" bank by convention
            if self.num_banks != 4:
                raise ValueError("Big-bank arrangement assumes 4 banks.")
            customers_per_bank = [20, 10, 5, 5]
        else:
            raise ValueError(f"Unknown arrangement: {self.arrangement}")

        bank_ids = list(self.banks.keys())
        depositor_id = 0
        type_index = 0
        for bank_id, n_cust in zip(bank_ids, customers_per_bank):
            for _ in range(n_cust):
                depositor_type = types[type_index]
                type_index += 1
                dep = Depositor(
                    unique_id=depositor_id,
                    bank_id=bank_id,
                    depositor_type=depositor_type,
                    initial_deposit=self.deposit_per_depositor,
                )
                self.depositors[depositor_id] = dep
                self.banks[bank_id].add_customer(depositor_id)
                depositor_id += 1

        # Now update each bank's initial deposits and reserves
        for bank in self.banks.values():
            num_customers = len(bank.customers)
            total_deposits = num_customers * self.deposit_per_depositor
            bank.initial_deposits = float(total_deposits)
            bank.reserves = self.reserve_ratio * bank.initial_deposits
            bank.illiquid = (1.0 - self.reserve_ratio) * bank.initial_deposits

    def _create_interbank_network(self):
        """
        Create the interbank market as an undirected graph using networkx.

        - no_interbank:  no edges
        - symmetric:     fully connected (complete graph)
        - big_bank:      star network centered on bank 0 (the big bank)
        """
        self.interbank_graph = nx.Graph()
        self.interbank_graph.add_nodes_from(self.banks.keys())

        if not self.has_interbank:
            return

        if self.arrangement == "symmetric":
            # Fully connected
            for i in self.banks.keys():
                for j in self.banks.keys():
                    if i < j:
                        self.interbank_graph.add_edge(i, j)

        elif self.arrangement == "big_bank":
            # Star centered on bank 0
            big_id = 0
            for j in self.banks.keys():
                if j != big_id:
                    self.interbank_graph.add_edge(big_id, j)

        else:
            raise ValueError(f"Unknown arrangement: {self.arrangement}")

    # ---------- Interbank borrowing and failure propagation ----------

    def borrow_from_interbank(self, borrower_bank, shortage):
        """
        Simple interbank borrowing rule:
        - Borrower contacts its neighbors in random order.
        - Each neighbor lends up to its excess reserves above a (here zero) safety buffer.
        - We record interbank assets and liabilities so that failures propagate.
        """
        if shortage <= 0:
            return 0.0

        neighbors = list(self.interbank_graph.neighbors(borrower_bank.bank_id))
        self._rng.shuffle(neighbors)

        total_borrowed = 0.0
        safety_buffer = 0.0  # could be > 0 to limit contagion

        for nb_id in neighbors:
            if total_borrowed >= shortage:
                break
            nb_bank = self.banks[nb_id]
            if nb_bank.failed:
                continue
            excess = max(0.0, nb_bank.reserves - safety_buffer)
            if excess <= 0:
                continue

            lend = min(excess, shortage - total_borrowed)
            if lend <= 0:
                continue

            # Lender gives reserves
            nb_bank.reserves -= lend
            total_borrowed += lend

            # Record interbank positions
            nb_bank.interbank_assets[borrower_bank.bank_id] = (
                nb_bank.interbank_assets.get(borrower_bank.bank_id, 0.0) + lend
            )
            borrower_bank.interbank_liabilities[nb_id] = (
                borrower_bank.interbank_liabilities.get(nb_id, 0.0) + lend
            )

        return total_borrowed

    def _wipe_depositors(self, bank):
        """
        When a bank fails, its remaining depositors lose their balances.
        """
        for dep_id in bank.customers:
            dep = self.depositors[dep_id]
            if dep.balance > 0:
                dep.balance = 0.0

    def _trigger_secondary_failure(self, bank):
        """
        Failure of a bank induced by interbank losses.
        """
        if bank.failed:
            return
        bank.failed = True
        if bank.bank_id not in self.bank_failures:
            self.bank_failures.append(bank.bank_id)
        # For secondary failures we do not change first_failure_step (that is
        # defined by the initial failure).
        self._wipe_depositors(bank)
        self._propagate_interbank_losses(bank.bank_id)

    def _propagate_interbank_losses(self, failed_bank_id):
        """
        When bank 'failed_bank_id' fails, every other bank that holds an asset
        (loan) on it loses that asset. This may make the creditor insolvent and
        cause a cascade of failures.
        """
        for creditor in self.banks.values():
            if creditor.failed:
                continue
            if failed_bank_id in creditor.interbank_assets:
                loss = creditor.interbank_assets.pop(failed_bank_id)
                creditor.reserves -= loss
                if creditor.reserves < 0:
                    self._trigger_secondary_failure(creditor)

    def register_bank_failure(self, bank):
        """
        Record first failure time, wipe depositors, and propagate interbank losses.
        """
        if bank.bank_id not in self.bank_failures:
            self.bank_failures.append(bank.bank_id)
        if self.first_failure_step is None:
            self.first_failure_step = self.current_step

        # Wipe its depositors
        self._wipe_depositors(bank)

        # Propagate interbank losses
        self._propagate_interbank_losses(bank.bank_id)

    # ---------- Mesa Model API ----------

    def step(self):
        """
        Advance the model by one period: each bank serves at most one customer.
        """
        if self.current_step == 0:
            self.datacollector.collect(self)

        self.current_step += 1
        self.schedule.step()
        self.datacollector.collect(self)

    def run_model(self):
        """
        Run until max_steps or until all banks have served all customers
        (or failed).
        """
        while self.current_step < self.max_steps:
            # Check if all banks are done serving customers (no queues left)
            all_done = all(
                (bank.queue_index >= len(bank.customers)) or bank.failed
                for bank in self.banks.values()
            )
            if all_done:
                break
            self.step()
