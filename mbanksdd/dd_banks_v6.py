# mbanksdd/dd_banks_v6.py
#
# Queue-based Mesa implementation of the multibank Diamond–Dybvig ABM (V6),
# designed to preserve the key economic logic from Pedro Romero's NetLogo model:
#
#  - Depositors choose early vs late withdrawal based on type, Rc, Rb & queue info
#  - Banks invest leftover funds at rate Rb; when withdrawals are large, this can
#    *reduce* funds (because (init_deposits - tot_withdrawals) can be negative)
#  - Banks face runs when they hit fin_balance < 0 while there are still
#    unserved depositors in their queue (n_served < num_customers)
#  - Industrial setups:
#       * Isolated banks (no interbank market)
#       * Interbank market with symmetric banks
#       * Interbank market with one "big" bank
#
# The model is written in a Mesa style but does NOT subclass mesa.Agent
# to avoid version issues; only Model + DataCollector come from Mesa.

from mesa import Model
from mesa.datacollection import DataCollector
import random
from typing import List, Optional, Dict, Set


# ----------------------------------------------------------------------
# Customer / Depositor
# ----------------------------------------------------------------------

class Customer:
    """
    Depositor (NetLogo 'customers').

    State:
      - deposit_t0 : current deposit
      - withdraw1, withdraw2 : potential t=1, t=2 withdrawals (decision)
      - Rc       : subjective return
      - account1, account2 : post-withdrawal accounts (for diagnostics)
      - dtype    : "impatient" or "patient"
      - active   : can still be served / has funds
      - has_decided : has already chosen withdrawals (served in queue)
    """

    def __init__(self, unique_id: str, model: "BankRunModelV6",
                 bank: "Bank", dtype: str, initial_deposit: float = 1.0):
        self.unique_id = unique_id
        self.model = model
        self.bank = bank
        self.dtype = dtype

        self.deposit_t0 = initial_deposit

        self.withdraw1 = 0.0
        self.withdraw2 = 0.0

        self.Rc = 1.0
        self.account1 = initial_deposit
        self.account2 = initial_deposit

        self.fitness1 = 0.0
        self.fitness2 = 0.0

        self.active = True
        self.has_decided = False

    # --- decision primitives -------------------------------------------------

    def draw_subjective_rate(self):
        """Draw Rc ~ 1 + U(0, 0.5) each period."""
        if not self.active:
            return
        self.Rc = 1.0 + random.random() * 0.5

    def decide_withdrawals(self):
        """
        Early vs late withdrawal decision (NetLogo 'make-decision', simplified).

        We DO NOT apply withdrawals here; we just decide withdraw1/withdraw2.
        Deposits / accounts are updated later, *after* banks update balances,
        so that even depositors who end up with negative accounts still count
        in tot-withdrawals in this step (as in NetLogo).
        """
        if not self.active or self.has_decided:
            return

        bank = self.bank
        Rb = bank.Rb

        # Small random consumption preference for early withdrawal
        consume1 = 1.0 + random.random() * 0.2

        # Bank-level queue information
        bank_frac_served = bank.n_served / max(1, bank.num_customers)

        # System-wide impatience information (NetLogo spirit)
        # Originally: (n_served/10) vs (n_impatient/(40*impatient_prob)).
        # Here we generalize to arbitrary bank sizes:
        #   threshold ≈ "fraction of customers in system who are impatient".
        if self.model.impatient_probability > 0:
            system_impatient_fraction = (
                self.model.n_impatient_total /
                (self.model.num_customers_total * self.model.impatient_probability)
            )
        else:
            system_impatient_fraction = 1.0  # degenerate case

        # --- Impatient agents -------------------------------------------------
        if self.dtype == "impatient":
            # If lots of people have already been served relative to expected impatient mass,
            # the impatient may "rush" and grab 1 unit (like your NetLogo).
            if bank_frac_served > system_impatient_fraction:
                self.fitness1 = 1.0
            else:
                self.fitness1 = consume1

            self.withdraw1 = self.fitness1
            self.fitness2 = 0.0
            self.withdraw2 = 0.0

        # --- Patient agents ---------------------------------------------------
        else:
            # Condition to trigger early withdrawal for patient agents:
            # many already served at this bank AND Rc > Rb (run-like behavior)
            if (bank_frac_served > system_impatient_fraction) and (self.Rc > Rb):
                self.fitness1 = consume1
                self.withdraw1 = self.fitness1
                self.fitness2 = 0.0
                self.withdraw2 = 0.0
            else:
                # Late withdrawal: NetLogo's functional form
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

        # Mark that this customer has been processed in the queue
        self.has_decided = True

    def apply_withdrawals_and_update_accounts(self):
        """
        Apply withdraw1/withdraw2 to deposit_t0 and update account1/account2.
        This is the analogue of NetLogo 'fitness-check', but it is called
        AFTER banks have used withdraw1+withdraw2 to compute fin_balance.
        """
        if not self.active:
            return

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


# ----------------------------------------------------------------------
# Bank
# ----------------------------------------------------------------------

class Bank:
    """
    Bank (NetLogo 'banks').

    State:
      - init_deposits : sum of initial deposits of its customers
      - fin_balance   : current balance (after investment + withdrawals)
      - Rb            : bank's investment return
      - queue_index   : where we are in the service queue
      - served_customers : set of customers who have been served at least once
      - n_served      : |served_customers|
      - active, failed
    """

    def __init__(self, unique_id: str, model: "BankRunModelV6",
                 customer_ids: List[int]):
        self.unique_id = unique_id
        self.model = model
        self.customer_ids = customer_ids
        self.customers: List["Customer"] = []

        self.num_customers = len(customer_ids)

        self.init_deposits = self.num_customers * self.model.initial_deposit
        self.fin_balance = self.init_deposits

        self.Rb = 1.0

        # Queue state
        self.queue_index = 0
        self.served_customers: Set[str] = set()
        self.n_served = 0

        self.active = True
        self.failed = False

    # --- primitives ----------------------------------------------------------

    def draw_bank_rate(self):
        """Draw Rb ~ 1 + U(0, 0.5) each period."""
        if not self.active:
            return
        self.Rb = 1.0 + random.random() * 0.5

    def serve_queue(self, customers_per_step: int = 1):
        """
        Serve at most `customers_per_step` depositors from this bank's queue.
        Each served depositor decides withdraw1/withdraw2 ONCE (has_decided flag).
        """
        if not self.active:
            return

        served_this_step = 0

        while (served_this_step < customers_per_step) and (self.queue_index < len(self.customers)):
            cust = self.customers[self.queue_index]
            self.queue_index += 1

            if not cust.active or cust.has_decided:
                continue

            cust.decide_withdrawals()

            if cust.withdraw1 > 0 or cust.withdraw2 > 0:
                self.served_customers.add(cust.unique_id)

            served_this_step += 1

        self.n_served = len(self.served_customers)

    def update_balance_sheet(self):
        """
        Bank balance sheet update (NetLogo-style):

        1. Compute total withdrawals (everyone counts, even if they will later go negative).
        2. Decide investment based on fraction with withdraw1 vs expected impatients.
        3. Update fin_balance = init_deposits - tot_withdrawals + invest.
        4. If fin_balance < 0 and there are still customers in line (n_served < num_customers):
           - If interbank market: call interbank_support()
           - Else: go_bankrupt()
        """
        if not self.active:
            return

        # 1. Total withdrawals across all customers
        tot_withdrawals = 0.0
        n_withdraw1 = 0
        for c in self.customers:
            tot_withdrawals += (c.withdraw1 + c.withdraw2)
            if c.withdraw1 > 0:
                n_withdraw1 += 1

        # 2. Investment rule
        frac_withdraw1 = n_withdraw1 / max(1, self.num_customers)

        n_impatient = self.model.n_impatient_total
        expected_impatient = (
            self.model.impatient_probability * self.model.num_customers_total
            if self.model.impatient_probability > 0
            else 1.0
        )
        threshold = n_impatient / max(1e-9, expected_impatient)

        if frac_withdraw1 <= threshold:
            invest = (self.init_deposits - tot_withdrawals) * self.Rb
        else:
            invest = 0.0

        # 3. New balance before interbank support
        self.fin_balance = self.init_deposits - tot_withdrawals + invest

        # 4. Run condition: negative balance while queue not fully served
        if self.fin_balance < 0 and (self.n_served < self.num_customers):
            if self.model.interbank_market:
                self.interbank_support()
            else:
                self.go_bankrupt()

    def interbank_support(self):
        """
        System-wide pooling:

        - Compute this bank's deficit D = -fin_balance (>0).
        - Look at all other active banks with fin_balance > 0.
        - Let S = sum of their surpluses.
        - If S >= D: transfer D to this bank, taking from donors
          proportionally to their surpluses (donors stay >= 0).
        - If S < D: this bank still can't be saved -> go_bankrupt().
        """
        if not self.active:
            return

        deficit = -self.fin_balance
        if deficit <= 0:
            return

        donors = [
            b for b in self.model.banks
            if (b is not self) and b.active and (b.fin_balance > 0)
        ]

        total_surplus = sum(b.fin_balance for b in donors)

        if total_surplus <= 0:
            # No liquidity in the rest of the system
            self.go_bankrupt()
            return

        # Amount of support available (cannot exceed total_surplus)
        support = min(deficit, total_surplus)

        # Take proportionally from all donors
        for b in donors:
            share = b.fin_balance / total_surplus
            deduction = share * support
            b.fin_balance -= deduction

        # Give full support to this bank
        self.fin_balance += support

        # If still insolvent, fail
        if self.fin_balance < 0:
            self.go_bankrupt()

    def go_bankrupt(self):
        """Mark bank as failed (run)."""
        self.active = False
        self.failed = True



# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

class BankRunModelV6(Model):
    """
    Queue-based multibank Diamond–Dybvig ABM.

    Parameters
    ----------
    num_banks : int
        Number of banks.
    customer_distribution : list[int], optional
        Customers per bank; e.g. [10,10,10,10] or [10,5,5,5].
    interbank_market : bool
        If True, banks can borrow from each other.
    impatient_probability : float
        Probability a depositor is "impatient".
    initial_deposit : float
        Initial deposit per customer.
    customers_per_step : int
        How many customers per bank are served from the queue each step.
    seed : int, optional
        Random seed (for reproducibility). If None, do not reseed.
    """

    def __init__(
        self,
        num_banks: int = 4,
        customer_distribution: Optional[List[int]] = None,
        interbank_market: bool = True,
        impatient_probability: float = 0.5,
        initial_deposit: float = 1.0,
        customers_per_step: int = 1,
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
        self.customers_per_step = customers_per_step

        if customer_distribution is None:
            customer_distribution = [10] * num_banks
        assert len(customer_distribution) == num_banks
        self.customer_distribution = customer_distribution

        self.num_customers_total = sum(customer_distribution)
        self.time = 0

        self.banks: List[Bank] = []
        self.customers: List[Customer] = []

        # 1. Create banks
        customer_id_counter = 0
        for bank_id in range(num_banks):
            n_cust = customer_distribution[bank_id]
            customer_ids_for_bank = list(range(customer_id_counter,
                                               customer_id_counter + n_cust))
            customer_id_counter += n_cust
            bank = Bank(unique_id=f"B{bank_id}", model=self,
                        customer_ids=customer_ids_for_bank)
            self.banks.append(bank)

        # 2. Create customers and attach to banks
        for bank in self.banks:
            for cid in bank.customer_ids:
                if random.random() < self.impatient_probability:
                    dtype = "impatient"
                else:
                    dtype = "patient"

                cust = Customer(
                    unique_id=f"C{cid}",
                    model=self,
                    bank=bank,
                    dtype=dtype,
                    initial_deposit=self.initial_deposit,
                )
                self.customers.append(cust)
                bank.customers.append(cust)

        self.n_impatient_total = sum(1 for c in self.customers if c.dtype == "impatient")

        # DataCollector: for now, just bank failure count & average fin_balance
        self.datacollector = DataCollector(
            model_reporters={
                "num_failed_banks": lambda m: sum(1 for b in m.banks if b.failed),
                "avg_fin_balance": lambda m: (
                    sum(b.fin_balance for b in m.banks) / len(m.banks)
                    if m.banks else 0.0
                ),
            }
        )

    # ------------------------------------------------------------------
    # One step of the model
    # ------------------------------------------------------------------

    def step(self):
        """
        Sequence:
          1. Banks draw Rb; active customers draw Rc.
          2. Each bank serves a few customers in its queue (customers_per_step).
          3. Banks update balance sheets, may borrow or fail.
          4. Customers apply withdrawals to their deposits and may become inactive.
          5. Collect data.
        """
        # 1. Draw rates
        for bank in self.banks:
            bank.draw_bank_rate()

        for cust in self.customers:
            cust.draw_subjective_rate()

        # 2. Serve queue
        for bank in self.banks:
            bank.serve_queue(customers_per_step=self.customers_per_step)

        # 3. Bank balance sheets
        for bank in self.banks:
            bank.update_balance_sheet()

        # 4. Apply withdrawals to customer accounts
        for cust in self.customers:
            cust.apply_withdrawals_and_update_accounts()

        # 5. Collect data
        self.datacollector.collect(self)
        self.time += 1

    def run_model(self, n_steps: int):
        for _ in range(n_steps):
            self.step()


# ----------------------------------------------------------------------
# Helper constructors for the three industrial setups
# ----------------------------------------------------------------------

def make_isolated_model(
    num_banks: int = 4,
    customers_per_bank: int = 10,
    impatient_probability: float = 0.5,
    customers_per_step: int = 1,
    seed: Optional[int] = None,
) -> BankRunModelV6:
    """Isolated banks (no interbank market)."""
    distribution = [customers_per_bank] * num_banks
    return BankRunModelV6(
        num_banks=num_banks,
        customer_distribution=distribution,
        interbank_market=False,
        impatient_probability=impatient_probability,
        initial_deposit=1.0,
        customers_per_step=customers_per_step,
        seed=seed,
    )


def make_interbank_model(
    num_banks: int = 4,
    customers_per_bank: int = 10,
    impatient_probability: float = 0.5,
    customers_per_step: int = 1,
    seed: Optional[int] = None,
) -> BankRunModelV6:
    """Symmetric banks with an interbank market."""
    distribution = [customers_per_bank] * num_banks
    return BankRunModelV6(
        num_banks=num_banks,
        customer_distribution=distribution,
        interbank_market=True,
        impatient_probability=impatient_probability,
        initial_deposit=1.0,
        customers_per_step=customers_per_step,
        seed=seed,
    )


def make_big_bank_model(
    big_bank_index: int = 0,
    big_size: int = 25,
    small_size: int = 5,
    num_small_banks: int = 3,
    impatient_probability: float = 0.5,
    customers_per_step: int = 1,
    seed: Optional[int] = None,
) -> BankRunModelV6:
    """
    One big bank in an interbank market.
    Example: big_size=10, small_size=5, num_small_banks=3 -> [10,5,5,5].
    """
    customer_distribution: List[int] = []
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
        initial_deposit=1.0,
        customers_per_step=customers_per_step,
        seed=seed,
    )
