"""
Model 1: Diamond–Dybvig ABM with the ORIGINAL banking contract
(no social network, original late payoff based on r1 and the fraction
of early withdrawals).

Structure matches model.py (Model 3) but _decide_withdrawals implements
Model 1 behavior.
"""

from __future__ import annotations

from typing import List, Optional

from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector


# ---------------------------------------------------------------------
# Agents (plain Python classes, NOT subclassing mesa.Agent)
# ---------------------------------------------------------------------


class BankAgent:
    """
    Bank balance sheet.

    init_deposits : sum of all initial deposits
    fin_balance   : init_deposits minus total withdrawals
    served        : number of depositors that have ever withdrawn
    """

    def __init__(self, unique_id: str, model: "OneBankDDModel1") -> None:
        self.unique_id = unique_id
        self.model = model
        self.random = model.random
        self.pos = None  # MultiGrid will set this

        self.init_deposits: float = 0.0
        self.fin_balance: float = 0.0
        self.served: int = 0

    def step(self) -> None:
        # Not used, but kept for symmetry with Mesa agents.
        return


class DepositorAgent:
    """
    Depositor for Model 1 (original DD contract, no social network).

    impatient : True  -> impatient (green)
                False -> patient   (yellow)
    deposit_t0 : remaining deposit at the bank
    withdraw1, withdraw2 : early and late withdrawals
    r1, R : gross returns for early and late withdrawal
    fitness1, fitness2 : payoffs associated with early and late positions
    active : when False, agent stops making decisions
    served : whether this depositor has been served in the queue
    """

    def __init__(
        self,
        unique_id: int,
        model: "OneBankDDModel1",
        impatient: bool,
        initial_deposit: float = 1.0,
    ) -> None:
        self.unique_id = unique_id
        self.model = model
        self.random = model.random
        self.pos = None  # MultiGrid will set this

        self.impatient: bool = impatient

        # economic state
        self.deposit_t0: float = initial_deposit
        self.withdraw1: float = 0.0
        self.withdraw2: float = 0.0
        self.r1: float = 1.2
        self.R: float = 2.0
        self.fitness1: float = 0.0
        self.fitness2: float = 0.0

        self.active: bool = True
        self.served: bool = False

    # ---------- helpers corresponding to NetLogo procedures ----------

    def _set_rates(self) -> None:
        """
        NetLogo 'set-rates':

        - fixed   -> R = 2, r1 = 1.2
        - random  -> R ~ U(1.2, 2.0), r1 ~ U(1.0, 1.2)
        """
        if self.model.rates_mode == "fixed":
            self.R = 2.0
            self.r1 = 1.2
        else:
            self.R = 1.2 + self.random.random() * 0.8
            self.r1 = 1.0 + self.random.random() * 0.2

    def _decide_withdrawals(self) -> None:
        """
        MODEL 1 DECISION RULE
        ---------------------
        - Original DD-style contract:
            Early payoff is based on r1.
            Late payoff:
                V2 = R * (1 - r1 * phi) / (1 - phi),
            where phi is the fraction of agents who withdraw early.
        - No social network.
        - 'consumption_mode':
            * 'constant'  -> withdraw all payoff (w = 1)
            * 'variable'  -> withdraw w in (0,1) of the payoff
        """
        if not self.active or self.served:
            return

        # Global environment
        queuej = self.model.n_served
        n_impatient = self.model.num_impatients
        N = self.model.num_depositors
        n_withdraw1 = self.model.num_withdraw1

        # Random consumption draw (only used if "variable")
        if self.model.consumption_mode == "constant":
            w = 1.0
        else:
            w = self.random.random()  # w in (0,1)

        # Fraction who have already withdrawn early
        frac_withdraw1 = n_withdraw1 / float(N) if N > 0 else 0.0

        # Late payoff V2 (original DD contract)
        V2 = 0.0
        if frac_withdraw1 < 1.0:
            V2 = self.R * (1.0 - (self.r1 * frac_withdraw1)) / (1.0 - frac_withdraw1)

        # Early payoff per unit deposit.
        # If the queue is longer than the number of impatient agents,
        # force early payoff down to 1 (bank stress).
        early_base = self.r1
        if queuej > n_impatient:
            early_base = 1.0

        # Reset
        self.withdraw1 = 0.0
        self.withdraw2 = 0.0
        self.fitness1 = 0.0
        self.fitness2 = 0.0

        if self.impatient:
            # Impatient agents always withdraw early
            self.fitness1 = early_base
            self.withdraw1 = w * self.fitness1
        else:
            # Patient agents: compare early vs late payoff (no network).
            if early_base >= V2:
                self.fitness1 = early_base
                self.withdraw1 = w * self.fitness1
            else:
                self.fitness2 = V2
                self.withdraw2 = w * self.fitness2

        # Update remaining deposit and activity status
        total_withdrawn = self.withdraw1 + self.withdraw2
        self.deposit_t0 = max(self.deposit_t0 - total_withdrawn, 0.0)
        if self.deposit_t0 <= 0.0:
            self.active = False

    def serve(self) -> None:
        """
        Serve this depositor once in the sequential queue.
        """
        if not self.active or self.served:
            self.served = True
            return

        self._set_rates()
        self._decide_withdrawals()
        self.served = True


# ---------------------------------------------------------------------
# Model 1
# ---------------------------------------------------------------------


class OneBankDDModel1(Model):
    """
    One-bank Diamond–Dybvig ABM – Model 1 (original contract, no network).

    Parameters
    ----------
    width, height : grid dimensions (default 21x21 -> 441 depositors)
    impatient_probability : probability a depositor is impatient
    rates_mode : 'fixed' or 'random'
    consumption_mode : 'constant' or 'variable'
    initial_deposit : initial deposit per depositor
    seed : random seed (optional)
    """

    def __init__(
        self,
        width: int = 21,
        height: int = 21,
        impatient_probability: float = 0.5,
        rates_mode: str = "fixed",
        consumption_mode: str = "variable",
        initial_deposit: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        if seed is not None:
            self.seed = seed
            self.random.seed(seed)

        self.width = width
        self.height = height
        self.impatient_probability = impatient_probability
        self.rates_mode = rates_mode
        self.consumption_mode = consumption_mode
        self.initial_deposit = initial_deposit

        # space
        self.grid = MultiGrid(width, height, torus=True)

        # bank at center
        self.bank = BankAgent("bank", self)
        center = (width // 2, height // 2)
        self.grid.place_agent(self.bank, center)

        # one depositor per patch
        self.depositors: List[DepositorAgent] = []
        uid = 0
        for x in range(width):
            for y in range(height):
                impatient = self.random.random() <= self.impatient_probability
                d = DepositorAgent(
                    unique_id=uid,
                    model=self,
                    impatient=impatient,
                    initial_deposit=self.initial_deposit,
                )
                self.depositors.append(d)
                self.grid.place_agent(d, (x, y))
                uid += 1

        self.num_depositors: int = len(self.depositors)

        # initial bank balance
        self.bank.init_deposits = sum(d.deposit_t0 for d in self.depositors)
        self.bank.fin_balance = self.bank.init_deposits
        self.bank.served = 0

        # sequential service queue
        self._service_queue: List[DepositorAgent] = self.random.sample(
            self.depositors, k=self.num_depositors
        )
        self._queue_index: int = 0

        # data collection: same reporters as Model 3
        self.datacollector = DataCollector(
            model_reporters={
                "deposits": lambda m: m.bank.fin_balance,
                "totalw": lambda m: m.total_withdrawals,
                "type1w": lambda m: m.total_withdraw1,
                "type2w": lambda m: m.total_withdraw2,
                "withdrew": lambda m: m.total_withdrawals,
                "patients": lambda m: m.num_patients,
                "impatients": lambda m: m.num_impatients,
                "served": lambda m: m.n_served,
                "bank_failed": lambda m: int(m.bank.fin_balance < 0.0),
            }
        )

        self.datacollector.collect(self)
        self.running = True

    # --------- derived quantities (NetLogo reporters) -----------------

    @property
    def num_impatients(self) -> int:
        return sum(1 for d in self.depositors if d.impatient)

    @property
    def num_patients(self) -> int:
        return self.num_depositors - self.num_impatients

    @property
    def n_served(self) -> int:
        return sum(
            1 for d in self.depositors if (d.withdraw1 > 0.0 or d.withdraw2 > 0.0)
        )

    @property
    def total_withdrawals(self) -> float:
        return sum(d.withdraw1 + d.withdraw2 for d in self.depositors)

    @property
    def total_withdraw1(self) -> float:
        return sum(d.withdraw1 for d in self.depositors)

    @property
    def total_withdraw2(self) -> float:
        return sum(d.withdraw2 for d in self.depositors)

    @property
    def num_withdraw1(self) -> int:
        return sum(1 for d in self.depositors if d.withdraw1 > 0.0)

    # --------- bank balance (bank-balance-sheet, invest off) ---------

    def _update_bank_balance(self) -> None:
        tot_withdrawals = self.total_withdrawals
        # Same as in your model: invest off (no R reinvestment)
        self.bank.fin_balance = self.bank.init_deposits - tot_withdrawals
        self.bank.served = self.n_served

    # --------- main evolution ----------------------------------------

    def step(self) -> None:
        if self._queue_index >= len(self._service_queue):
            self.running = False
            return

        agent = self._service_queue[self._queue_index]
        self._queue_index += 1
        agent.serve()

        self._update_bank_balance()
        self.datacollector.collect(self)

        if self.bank.fin_balance < 0.0:
            self.running = False

    def run_model(self, max_steps: Optional[int] = None) -> None:
        steps = 0
        while self.running and (max_steps is None or steps < max_steps):
            self.step()
            steps += 1


__all__ = ["OneBankDDModel1", "DepositorAgent", "BankAgent"]
