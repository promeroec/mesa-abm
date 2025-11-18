from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np


# -------------------------------------------------------------------
# Depositor (simple object, not a Mesa Agent)
# -------------------------------------------------------------------

class Depositor:
    """
    Simple depositor object (not a Mesa Agent).

    Attributes
    ----------
    id : int
        Unique depositor id (global).
    bank_id : int
        Id of the bank where this depositor holds a deposit.
    type : str
        "patient" or "impatient".
    balance : float
        Current deposit balance.
    has_withdrawn : bool
        True if depositor has already attempted / completed withdrawal.
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

        - Impatient depositors always withdraw when they are served.
        - Patient depositors withdraw (run) if, among those before them
          in the queue at this bank, the fraction that has already withdrawn
          is above model.panic_threshold.
        """
        if self.has_withdrawn or self.balance <= 0 or bank.failed:
            return False

        # Impatient depositors always withdraw
        if self.type == "impatient":
            return True

        # Patient depositor: look at previous positions in the queue
        num_before = position_in_queue
        if num_before <= 0:
            return False

        withdrawn_before = sum(
            1 for dep_id in bank.customers[:position_in_queue]
            if model.depositors[dep_id].has_withdrawn
        )
        frac_withdrawn = withdrawn_before / float(num_before)
        return frac_withdrawn >= model.panic_threshold


# -------------------------------------------------------------------
# Bank Agent
# -------------------------------------------------------------------

class BankAgent(Agent):
    """
    Bank agent implementing a Diamond–Dybvig style contract with reserves
    and access to an interbank market (no central bank here).

    Attributes
    ----------
    initial_deposits : float
        Total deposits at t=0.
    reserves : float
        Liquid asset available to meet withdrawals.
    illiquid : float
        Illiquid asset (long-term technology).
    failed : bool
        True if the bank has failed (cannot meet withdrawals).
    customers : list[int]
        List of depositor ids (queue order).
    queue_index : int
        Index of next depositor in the queue to be served.
    interbank_assets : dict[int, float]
        Amounts lent to other banks.
    interbank_liabilities : dict[int, float]
        Amounts borrowed from other banks.
    """

    def __init__(self, unique_id, model, initial_deposits, reserve_ratio):
        super().__init__(unique_id, model)
        self.initial_deposits = float(initial_deposits)
        self.reserve_ratio = float(reserve_ratio)
        self.reserves = self.reserve_ratio * self.initial_deposits
        self.illiquid = (1.0 - self.reserve_ratio) * self.initial_deposits
        self.failed = False

        self.customers = []
        self.queue_index = 0

        self.interbank_assets = {}      # {counterparty_id: amount}
        self.interbank_liabilities = {} # {counterparty_id: amount}

    def add_customer(self, depositor_id):
        self.customers.append(depositor_id)

    @property
    def bank_id(self):
        return self.unique_id

    def step(self):
        """
        One period: serve at most one depositor in the queue.
        """
        if self.failed:
            return
        if self.queue_index >= len(self.customers):
            return

        position_in_queue = self.queue_index
        dep_id = self.customers[position_in_queue]
        depositor = self.model.depositors[dep_id]

        if depositor.wants_to_withdraw(self, self.model, position_in_queue):
            self.process_withdrawal(depositor)

        self.queue_index += 1

    def process_withdrawal(self, depositor):
        """
        Attempt to pay out the depositor.

        First use reserves; if insufficient, attempt to borrow in interbank
        market; if still insufficient, the bank fails.
        """
        amount = depositor.balance
        if amount <= 0:
            return

        # Pay directly from reserves
        if self.reserves >= amount:
            self.reserves -= amount
            depositor.balance = 0.0
            depositor.has_withdrawn = True
            return

        # Use interbank market if available
        shortage = amount - self.reserves
        borrowed = 0.0
        if self.model.has_interbank:
            borrowed = self.model.borrow_from_interbank(self, shortage)

        total_liquidity = self.reserves + borrowed
        if total_liquidity >= amount:
            # Survives this withdrawal
            self.reserves = total_liquidity - amount
            depositor.balance = 0.0
            depositor.has_withdrawn = True
        else:
            # Cannot meet withdrawal: failure
            self.failed = True
            self.model.register_bank_failure(self)


# -------------------------------------------------------------------
# MultiBankModel
# -------------------------------------------------------------------

class MultiBankModel(Model):
    """
    Multi-bank Diamond–Dybvig style ABM based on Romero (2009), without a CB.

    Arrangements (only industrial organization differs, parameters same):

    - "no_interbank":
        4 banks, 40 depositors, 10 per bank, no interbank links.

    - "symmetric":
        4 banks, 40 depositors, 10 per bank, fully connected interbank graph.

    - "big_bank":
        4 banks, 40 depositors,
        bank 0: 20 customers, banks 1–3: 10, 5, 5 customers,
        star graph centered at bank 0.

    All cases:
        total_depositors = 40, num_patient + num_impatient = 40.
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

        # Random generator
        self._rng = np.random.default_rng(seed)

        # Parameters
        self.arrangement = arrangement
        self.num_banks = num_banks
        self.total_depositors = total_depositors
        self.num_patient = num_patient
        self.num_impatient = num_impatient
        self.deposit_per_depositor = deposit_per_depositor
        self.reserve_ratio = reserve_ratio
        self.panic_threshold = panic_threshold
        self.max_steps = max_steps

        if self.num_patient + self.num_impatient != self.total_depositors:
            raise ValueError("num_patient + num_impatient must equal total_depositors=40.")

        if self.total_depositors != 40:
            raise ValueError("This implementation assumes total_depositors=40 for now.")

        self.has_interbank = arrangement in ("symmetric", "big_bank")
        self.current_step = 0
        self.bank_failures = []
        self.first_failure_step = None

        # Scheduler: random order of banks each step
        self.schedule = RandomActivation(self)

        # Containers
        self.banks = {}
        self.depositors = {}

        # Build environment
        self._create_banks()
        self._create_depositors_and_assign()
        self._shuffle_customer_queues()
        self._create_interbank_network()

        # Data collector for summary stats
        self.datacollector = DataCollector(
            model_reporters={
                "num_failed_banks": lambda m: len(m.bank_failures),
                "first_failure_step": lambda m: m.first_failure_step
                if m.first_failure_step is not None else -1,
                "num_patient": lambda m: m.num_patient,
                "num_impatient": lambda m: m.num_impatient,
            }
        )

    # ---------------------- Construction helpers ----------------------

    def _create_banks(self):
        """
        Create bank agents, initially with zero deposits (set later).
        """
        for bank_id in range(self.num_banks):
            bank = BankAgent(bank_id, self, initial_deposits=0.0,
                             reserve_ratio=self.reserve_ratio)
            self.banks[bank_id] = bank
            self.schedule.add(bank)

    def _create_depositors_and_assign(self):
        """
        Create depositors and assign them to banks depending on arrangement.

        After assignment, set each bank's deposit base and reserves.
        """
        # Types vector: same across arrangements
        types = ["patient"] * self.num_patient + ["impatient"] * self.num_impatient
        self._rng.shuffle(types)

        # Customer allocation by arrangement
        if self.arrangement in ("no_interbank", "symmetric"):
            customers_per_bank = [10, 10, 10, 10]  # 4 × 10 = 40
        elif self.arrangement == "big_bank":
            if self.num_banks != 4:
                raise ValueError("big_bank arrangement assumes num_banks=4.")
            customers_per_bank = [20, 10, 5, 5]   # big bank + 3 small = 40
        else:
            raise ValueError(f"Unknown arrangement: {self.arrangement}")

        depositor_id = 0
        type_index = 0
        for bank_id, n_cust in enumerate(customers_per_bank):
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

        # Now compute each bank's deposit base and reserves
        for bank in self.banks.values():
            num_customers = len(bank.customers)
            total_deposits = num_customers * self.deposit_per_depositor
            bank.initial_deposits = float(total_deposits)
            bank.reserves = self.reserve_ratio * bank.initial_deposits
            bank.illiquid = (1.0 - self.reserve_ratio) * bank.initial_deposits

    def _shuffle_customer_queues(self):
        """
        Randomize the queue order of customers within each bank.
        """
        for bank in self.banks.values():
            self._rng.shuffle(bank.customers)

    def _create_interbank_network(self):
        """
        Build the interbank network:

        - no_interbank: empty graph
        - symmetric: fully connected graph
        - big_bank: star centered at bank 0
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
            # Star centered at bank 0
            center = 0
            for j in self.banks.keys():
                if j != center:
                    self.interbank_graph.add_edge(center, j)
        else:
            raise ValueError(f"Unknown arrangement: {self.arrangement}")

    # ---------------- Interbank borrowing & contagion -----------------

    def borrow_from_interbank(self, borrower_bank, shortage):
        """
        Borrow from neighbors in random order until the shortage is filled
        or no more reserves are available.

        We track interbank_assets/liabilities so that failures propagate
        losses to creditors.
        """
        if shortage <= 0:
            return 0.0

        neighbors = list(self.interbank_graph.neighbors(borrower_bank.bank_id))
        self._rng.shuffle(neighbors)

        total_borrowed = 0.0
        for nb_id in neighbors:
            if total_borrowed >= shortage:
                break
            nb_bank = self.banks[nb_id]
            if nb_bank.failed:
                continue

            excess = max(0.0, nb_bank.reserves)
            if excess <= 0:
                continue

            lend = min(excess, shortage - total_borrowed)
            if lend <= 0:
                continue

            nb_bank.reserves -= lend
            total_borrowed += lend

            # Record bilateral positions
            nb_bank.interbank_assets[borrower_bank.bank_id] = \
                nb_bank.interbank_assets.get(borrower_bank.bank_id, 0.0) + lend
            borrower_bank.interbank_liabilities[nb_id] = \
                borrower_bank.interbank_liabilities.get(nb_id, 0.0) + lend

        return total_borrowed

    def _wipe_depositors(self, bank):
        """
        On failure, all remaining deposits at this bank are wiped out.
        """
        for dep_id in bank.customers:
            dep = self.depositors[dep_id]
            dep.balance = 0.0

    def _trigger_secondary_failure(self, bank):
        """
        Trigger secondary failure from interbank losses.
        """
        if bank.failed:
            return

        bank.failed = True
        if bank.bank_id not in self.bank_failures:
            self.bank_failures.append(bank.bank_id)

        # Do not change first_failure_step here: that is defined by
        # the earliest failure; we only add contagion.
        self._wipe_depositors(bank)
        self._propagate_interbank_losses(bank.bank_id)

    def _propagate_interbank_losses(self, failed_bank_id):
        """
        When bank `failed_bank_id` fails, all banks that have an asset
        (loan) on it lose that asset. This can trigger cascades.
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
        Register a primary bank failure, wipe its depositors, and propagate
        interbank losses to creditors.
        """
        if bank.bank_id not in self.bank_failures:
            self.bank_failures.append(bank.bank_id)
        if self.first_failure_step is None:
            self.first_failure_step = self.current_step

        self._wipe_depositors(bank)
        self._propagate_interbank_losses(bank.bank_id)

    # ---------------------- Model API ----------------------

    def step(self):
        """
        Advance the model by one time step: each bank serves one customer.
        """
        if self.current_step == 0:
            self.datacollector.collect(self)

        self.current_step += 1
        self.schedule.step()
        self.datacollector.collect(self)

    def run_model(self):
        """
        Run model until max_steps or until all banks are either done
        serving customers or have failed.
        """
        while self.current_step < self.max_steps:
            all_done = all(
                (bank.queue_index >= len(bank.customers)) or bank.failed
                for bank in self.banks.values()
            )
            if all_done:
                break
            self.step()
