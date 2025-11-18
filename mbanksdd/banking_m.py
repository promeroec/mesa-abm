from mesa import Agent, Model
from mesa.time import RandomActivation
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
        Withdrawal decision rule:
        - Impatient depositors always withdraw.
        - Patient depositors withdraw if panic threshold is exceeded.
        """
        if self.has_withdrawn or self.balance <= 0 or bank.failed:
            return False

        if self.type == "impatient":
            return True

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
    def __init__(self, unique_id, model, initial_deposits, reserve_ratio):
        super().__init__(unique_id, model)
        self.initial_deposits = float(initial_deposits)
        self.reserve_ratio = float(reserve_ratio)
        self.reserves = self.reserve_ratio * self.initial_deposits
        self.illiquid = (1.0 - self.reserve_ratio) * self.initial_deposits
        self.failed = False
        self.customers = []
        self.queue_index = 0
        self.interbank_assets = {}
        self.interbank_liabilities = {}

    def add_customer(self, depositor_id):
        self.customers.append(depositor_id)

    @property
    def bank_id(self):
        return self.unique_id

    def step(self):
        if self.failed:
            return
        if self.queue_index >= len(self.customers):
            return

        dep_id = self.customers[self.queue_index]
        depositor = self.model.depositors[dep_id]

        if depositor.wants_to_withdraw(self, self.model, self.queue_index):
            self.process_withdrawal(depositor)

        self.queue_index += 1

    def process_withdrawal(self, depositor):
        amount = depositor.balance
        if amount <= 0:
            return

        if self.reserves >= amount:
            self.reserves -= amount
            depositor.balance = 0.0
            depositor.has_withdrawn = True
            return

        shortage = amount - self.reserves
        borrowed = 0.0
        if self.model.has_interbank:
            borrowed = self.model.borrow_from_interbank(self, shortage)

        total_liquidity = self.reserves + borrowed
        if total_liquidity >= amount:
            self.reserves = total_liquidity - amount
            depositor.balance = 0.0
            depositor.has_withdrawn = True
        else:
            self.failed = True
            self.model.register_bank_failure(self)

class MultiBankModel(Model):
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
        self._rng = np.random.default_rng(seed)
        self.arrangement = arrangement
        self.num_banks = num_banks
        self.total_depositors = total_depositors
        self.num_patient = num_patient
        self.num_impatient = num_impatient
        self.deposit_per_depositor = deposit_per_depositor
        self.reserve_ratio = reserve_ratio
        self.panic_threshold = panic_threshold
        self.max_steps = max_steps
        self.has_interbank = arrangement in ("symmetric", "big_bank")
        self.current_step = 0
        self.bank_failures = []
        self.first_failure_step = None

        self.schedule = RandomActivation(self)
        self.banks = {}
        self.depositors = {}

        self._create_banks()
        self._create_depositors_and_assign()
        self._shuffle_customer_queues()
        self._create_interbank_network()

        self.datacollector = DataCollector(model_reporters={
            "num_failed_banks": lambda m: len(m.bank_failures),
            "first_failure_step": lambda m: m.first_failure_step if m.first_failure_step is not None else -1,
        })

    def _create_banks(self):
        for bank_id in range(self.num_banks):
            bank = BankAgent(bank_id, self, 0.0, self.reserve_ratio)
            self.banks[bank_id] = bank
            self.schedule.add(bank)

    def _create_depositors_and_assign(self):
        types = ["patient"] * self.num_patient + ["impatient"] * self.num_impatient
        self._rng.shuffle(types)

        if self.arrangement in ("no_interbank", "symmetric"):
            customers_per_bank = [10, 10, 10, 10]
        else:
            customers_per_bank = [20, 10, 5, 5]

        depositor_id = 0
        type_index = 0
        for bank_id, n_cust in enumerate(customers_per_bank):
            for _ in range(n_cust):
                depositor_type = types[type_index]
                type_index += 1
                dep = Depositor(depositor_id, bank_id, depositor_type, self.deposit_per_depositor)
                self.depositors[depositor_id] = dep
                self.banks[bank_id].add_customer(depositor_id)
                depositor_id += 1

        for bank in self.banks.values():
            num_customers = len(bank.customers)
            total_deposits = num_customers * self.deposit_per_depositor
            bank.initial_deposits = float(total_deposits)
            bank.reserves = self.reserve_ratio * bank.initial_deposits
            bank.illiquid = (1.0 - self.reserve_ratio) * bank.initial_deposits

    def _shuffle_customer_queues(self):
        for bank in self.banks.values():
            self._rng.shuffle(bank.customers)

    def _create_interbank_network(self):
        self.interbank_graph = nx.Graph()
        self.interbank_graph.add_nodes_from(self.banks.keys())

        if not self.has_interbank:
            return

        if self.arrangement == "symmetric":
            for i in self.banks.keys():
                for j in self.banks.keys():
                    if i < j:
                        self.interbank_graph.add_edge(i, j)
        else:  # big bank: star
            for j in self.banks.keys():
                if j != 0:
                    self.interbank_graph.add_edge(0, j)

    def borrow_from_interbank(self, borrower_bank, shortage):
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
            nb_bank.reserves -= lend
            total_borrowed += lend

            nb_bank.interbank_assets[borrower_bank.bank_id] = nb_bank.interbank_assets.get(borrower_bank.bank_id, 0.0) + lend
            borrower_bank.interbank_liabilities[nb_id] = borrower_bank.interbank_liabilities.get(nb_id, 0.0) + lend

        return total_borrowed

    def _wipe_depositors(self, bank):
        for dep_id in bank.customers:
            dep = self.depositors[dep_id]
            dep.balance = 0.0

    def _propagate_interbank_losses(self, failed_bank_id):
        for creditor in self.banks.values():
            if creditor.failed:
                continue
            if failed_bank_id in creditor.interbank_assets:
                loss = creditor.interbank_assets.pop(failed_bank_id)
                creditor.reserves -= loss
                if creditor.reserves <
