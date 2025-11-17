# mbanksdd/dd_banks_v6.py
#
# Mesa implementation of Pedro Romero's multibank Diamond–Dybvig ABM (V6).
# Scenarios:
#   1. multi-bank, no interbank market
#   2. multi-bank, with interbank market
#   3. multi-bank, with interbank market + one "big" bank
#
# Use the helper functions at the bottom (make_isolated_model, etc.)
# to build the different scenarios.

from mesa import Model, Agent
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
import random
from typing import List, Optional


class CustomerAgent(Agent):
    """
    Depositor/customer.
    NetLogo analogue: breed 'customers'

    Key state:
      deposit_t0  (deposit-t0)
      withdraw1, withdraw2
      Rc          (subjective interest rate)
      account1, account2
      dtype       ("impatient" or "patient")
      active      (bool)
    """

    def __init__(self, unique_id, model, bank, dtype: str, initial_deposit: float = 1.0):
        super().__init__(unique_id, model)
        self.bank = bank
        self.dtype = dtype
        self.deposit_t0 = initial_deposit

        self.withdraw1 = 0.0
        self.withdraw2 = 0.0

        self.Rc = 1.0
        self.account1 = 0.0
        self.account2 = 0.0
        self.fitness1 = 0.0
        self.fitness2 = 0.0

        self.active = True

    def draw_subjective_rate(self):
        """Draw subjective return Rc ~ 1 + U(0, 0.5) each period."""
        self.Rc = 1.0 + random.random() * 0.5

    def decide_withdrawals(self):
        """
        Decision rule for early vs late withdrawal (clean version of NetLogo 'make-decision').

        - Impatient agents always withdraw at t=1 (size depends on queue pressure).
        - Patient agents may withdraw early if queues are long and Rc > Rb;
          otherwise they wait and withdraw at t=2.
        """
        if not self.active or self.deposit_t0 <= 0:
            self.withdraw1 = 0.0
            self.withdraw2 = 0.0
            return

        bank = self.bank
        Rb = bank.Rb

        consume1 = 1.0 + random.random() * 0.2

        # Bank-level queue pressure:
        frac_served = bank.n_served / max(1, bank.num_customers)

        # System-wide information about impatients (mirroring V6 logic):
        n_impatient = self.model.n_impatient_total
        expected_impatient = self.model.impatient_probability * self.model.num_customers_total
        threshold = n_impatient / max(1e-9, expected_impatient)

        # ---------- Impatient agents ----------
        if self.dtype == "impatient":
            if frac_served > threshold:
                self.fitness1 = 1.0
            else:
                self.fitness1 = consume1

            self.withdraw1 = self.fitness1
            self.fitness2 = 0.0
            self.withdraw2 = 0.0

        # ---------- Patient agents ----------
        else:
            if (frac_served > threshold) and (self.Rc > Rb):
                # Early withdrawal
                self.fitness1 = consume1
                self.withdraw1 = self.fitness1
                self.fitness2 = 0.0
                self.withdraw2 = 0.0
            else:
                # Wait to t=2
                if bank.n_served > 1:
                    denom = 1.0 - bank.n_served
                    if abs(denom) < 1e-9:
                        self.fitness2 = Rb
                    else:
                        self.fitness2 = (Rb * (1.0 - (consume1 * bank.n_served))) / denom
                else:
                    self.fitness2 = Rb

                self.withdraw2 = self.fitness2
                self.fitness1 = 0.0
                self.withdraw1 = 0.0

        self._update_accounts()

    def _update_accounts(self):
        """Update accounts & deposits after choosing withdrawals (NetLogo 'fitness-check')."""
        w1 = self.withdraw1
        w2 = self.withdraw2

        if self.dtype == "impatient":
            self.account1 = self.deposit_t0 - w1 - w2
            self.deposit_t0 = self.account1
            self.account2 = self.deposit_t0
        else:
            self.account2 = self.deposit_t0 - w2 - w1
            self.deposit_t0 = self.account2
            self.account1 = self.deposit_t0

        if (self.account1 < 0) or (self.account2 < 0) or (self.deposit_t0 < 0):
            self.active = False
            self.deposit_t0 = min(self.deposit_t0, 0.0)


class BankAgent(Agent):
    """
    Bank.
    NetLogo analogue: breed 'banks'

    Key state:
      init_deposits
      fin_balance
      loan_principal, loan_lender
      Rb, n_served
      active, failed
    """

    def __init__(self, unique_id, model, customer_ids: List[int]):
        super().__init__(unique_id, model)
        self.customer_ids = customer_ids
        self.customers: List[CustomerAgent] = []

        self.num_customers = len(customer_ids)

        self.init_deposits = self.num_customers * self.model.initial_deposit
        self.fin_balance = self.init_deposits

        self.loan_principal = 0.0
        self.loan_lender: Optional["BankAgent"] = None

        self.Rb = 1.0
        self.active = True
        self.failed = False
        self.n_served = 0

    def draw_bank_rate(self):
        """Draw bank rate Rb ~ 1 + U(0, 0.5) each period."""
        self.Rb = 1.0 + random.random() * 0.5

    def repay_previous_loan(self):
        """Repay any outstanding one-period interbank loan."""
        if self.loan_lender is None or self.loan_principal <= 0 or not self.active:
            return

        amount = self.loan_principal * 1.0001
        self.fin_balance -= amount
        self.loan_lender.fin_balance += amount

        self.loan_principal = 0.0
        self.loan_lender = None

    def update_balance_sheet(self):
        """
        Compute withdrawals, investment, new balance, and handle interbank loan / failure.
        Mirrors NetLogo 'bank-balance-sheet' in structure.
        """
        if not self.active:
            return

        # Repay previous loan first
        self.repay_previous_loan()

        tot_withdrawals = 0.0
        n_withdraw1 = 0
        n_served = 0

        for c in self.customers:
            if not c.active:
                continue
            tot_withdrawals += (c.withdraw1 + c.withdraw2)
            if c.withdraw1 > 0:
                n_withdraw1 += 1
            if (c.withdraw1 > 0) or (c.withdraw2 > 0):
                n_served += 1

        self.n_served = n_served

        # Investment rule
        n_impatient = self.model.n_impatient_total
        expected_impatient = self.model.impatient_probability * self.model.num_customers_total
        threshold = n_impatient / max(1e-9, expected_impatient)

        frac_withdraw1 = n_withdraw1 / max(1, self.num_customers)

        if frac_withdraw1 <= threshold:
            invest = (self.init_deposits - tot_withdrawals) * self.Rb
        else:
            invest = 0.0

        self.fin_balance = self.init_deposits - tot_withdrawals + invest + self.loan_principal

        if self.fin_balance < 0 and (self.n_served < self.num_customers):
            if self.model.interbank_market:
                self.request_interbank_loan()
            else:
                self.go_bankrupt()

    def request_interbank_loan(self):
        """Borrow 10% of some solvent bank's balance if possible."""
        if not self.active:
            return

        candidates = [
            b for b in self.model.banks
            if (b is not self) and b.active and (b.fin_balance > 0)
        ]

        if not candidates:
            self.go_bankrupt()
            return

        lender = self.random.choice(candidates)
        loan_amount = 0.1 * lender.fin_balance

        lender.fin_balance -= loan_amount
        self.fin_balance += loan_amount

        self.loan_principal = loan_amount
        self.loan_lender = lender

        if self.fin_balance < 0:
            self.go_bankrupt()

    def go_bankrupt(self):
        """Mark bank as failed (bank run)."""
        self.active = False
        self.failed = True


class BankRunModelV6(Model):
    """
    Generic V6 model.

    Parameters
    ----------
    num_banks : int
    customer_distribution : list[int]
        Customers per bank; e.g. [10,10,10,10] or [10,5,5,5].
    interbank_market : bool
        If True, allow interbank loans; else isolated banks.
    impatient_probability : float
    initial_deposit : float
    seed : int or None
    """

    def __init__(
        self,
        num_banks: int = 4,
        customer_distribution: Optional[List[int]] = None,
        interbank_market: bool = True,
        impatient_probability: float = 0.5,
        initial_deposit: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if seed is not None:
            random.seed(seed)
            self._seed = seed

        self.num_banks = num_banks
        self.interbank_market = interbank_market
        self.impatient_probability = impatient_probability
        self.initial_deposit = initial_deposit

        if customer_distribution is None:
            customer_distribution = [10] * num_banks
        assert len(customer_distribution) == num_banks
        self.customer_distribution = customer_distribution

        self.num_customers_total = sum(customer_distribution)

        self.schedule = BaseScheduler(self)
        self.banks: List[BankAgent] = []
        self.customers: List[CustomerAgent] = []

        # Create banks
        customer_id_counter = 0
        for bank_id in range(num_banks):
            n_cust = customer_distribution[bank_id]
            customer_ids_for_bank = list(range(customer_id_counter,
                                               customer_id_counter + n_cust))
            customer_id_counter += n_cust
            bank = BankAgent(unique_id=f"B{bank_id}", model=self,
                             customer_ids=customer_ids_for_bank)
            self.banks.append(bank)
            self.schedule.add(bank)

        # Create customers
        for bank in self.banks:
            for cid in bank.customer_ids:
                if random.random() < self.impatient_probability:
                    dtype = "impatient"
                else:
                    dtype = "patient"

                cust = CustomerAgent(
                    unique_id=f"C{cid}",
                    model=self,
                    bank=bank,
                    dtype=dtype,
                    initial_deposit=self.initial_deposit,
                )
                self.customers.append(cust)
                bank.customers.append(cust)
                self.schedule.add(cust)

        self.n_impatient_total = sum(1 for c in self.customers if c.dtype == "impatient")

        self.datacollector = DataCollector(
            model_reporters={
                "num_failed_banks": lambda m: sum(1 for b in m.banks if b.failed),
            },
            agent_reporters={
                "fin_balance": lambda a: getattr(a, "fin_balance", None),
                "deposit_t0": lambda a: getattr(a, "deposit_t0", None),
                "dtype": lambda a: getattr(a, "dtype", None),
            },
        )

    def step(self):
        """
        One period:
          1. Draw Rb for each bank and Rc for each customer.
          2. Customers decide withdrawals.
          3. Banks update balance sheets.
          4. Collect data.
        """
        for bank in self.banks:
            if bank.active:
                bank.draw_bank_rate()

        for cust in self.customers:
            if cust.active:
                cust.draw_subjective_rate()

        for cust in self.customers:
            cust.decide_withdrawals()

        for bank in self.banks:
            bank.update_balance_sheet()

        self.datacollector.collect(self)
        self.schedule.time += 1

    def run_model(self, n_steps: int):
        for _ in range(n_steps):
            self.step()


# ----------------------------------------------------------------------
# Helper constructors for Colab / notebooks
# ----------------------------------------------------------------------

def make_isolated_model(
    num_banks: int = 4,
    customers_per_bank: int = 10,
    impatient_probability: float = 0.5,
    seed: Optional[int] = None,
) -> BankRunModelV6:
    """Isolated banks (no interbank market)."""
    distribution = [customers_per_bank] * num_banks
    return BankRunModelV6(
        num_banks=num_banks,
        customer_distribution=distribution,
        interbank_market=False,
        impatient_probability=impatient_probability,
        seed=seed,
    )


def make_interbank_model(
    num_banks: int = 4,
    customers_per_bank: int = 10,
    impatient_probability: float = 0.5,
    seed: Optional[int] = None,
) -> BankRunModelV6:
    """Symmetric banks, with interbank market."""
    distribution = [customers_per_bank] * num_banks
    return BankRunModelV6(
        num_banks=num_banks,
        customer_distribution=distribution,
        interbank_market=True,
        impatient_probability=impatient_probability,
        seed=seed,
    )


def make_big_bank_model(
    big_bank_index: int = 0,
    big_size: int = 10,
    small_size: int = 5,
    num_small_banks: int = 3,
    impatient_probability: float = 0.5,
    seed: Optional[int] = None,
) -> BankRunModelV6:
    """
    One big bank and several small ones, with interbank market.
    Example: big_size=10, small_size=5, num_small_banks=3 -> [10,5,5,5].
    """
    customer_distribution = []
    for i in range(num_small_banks + 1):
        if i == big_bank_index:
            customer_distribution.append(big_size)
        else:
            customer_distribution.append(small_size)

    return BankRunModelV6(
        num_banks=len(customer_distribution),
        customer_distribution=customer_distribution,
        interbank_market=True,
        impatient_probability=impatient_probability,
        seed=seed,
    )
