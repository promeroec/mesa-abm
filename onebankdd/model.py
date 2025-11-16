"""
Pedro P. Romero's one-bank Diamond–Dybvig Agent-based model based on
my paper "Bank runs, banking contracts, and social networks." 

Key features:
    - 1 bank at the center of a 21x21 torus (441 patches / depositors)
    - Each patch is a depositor, either impatient (green) or patient (yellow)
    - Two rate regimes: 'fixed' and 'random'
    - Two consumption regimes: 'constant' and 'variable'
    - Social network effect: patient depositors may join a run if at least
      three of their eight neighbors have already withdrawn early.

This code is designed to match the *behavior* and GUI of the original
NetLogo model (DDmodelV4).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector


# ---------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------


class BankAgent(Agent):
    """
    Bank balance sheet.

    In DDmodelV4, the bank keeps track of:
        init-deposits : sum of all initial deposits
        fin-balance   : init-deposits minus total withdrawals (+ possible invest)
        served        : number of depositors that have ever withdrawn
    """

    def __init__(self, unique_id: str, model: "OneBankDDModel") -> None:
        super().__init__(unique_id, model)
        self.init_deposits: float = 0.0
        self.fin_balance: float = 0.0
        self.served: int = 0

    def step(self) -> None:
        # Bank behavior is handled directly by the model; nothing to do here.
        return


class DepositorAgent(Agent):
    """
    Depositor modeled after a NetLogo patch in DDmodelV4. :contentReference[oaicite:3]{index=3}

    Attributes
    ----------
    impatient : bool
        True  -> NetLogo pcolor = green  (impatient)
        False -> NetLogo pcolor = yellow (patient)
    deposit_t0 : float
        Remaining deposit at the bank.
    withdraw1, withdraw2 : float
        Amounts withdrawn as "early" and "late" withdrawals.
    r1, R : float
        Gross rates of return at t=1 and t=2 (patch-own in NetLogo).
    fitness1, fitness2 : float
        Payoffs associated with early (type-1) and late (type-2) withdrawals.
    active : bool
        When False, the agent stops making decisions (mirrors 'active' in NetLogo).
    served : bool
        Whether this depositor has already been served in the sequential queue.
    """

    def __init__(
        self,
        unique_id: int,
        model: "OneBankDDModel",
        impatient: bool,
        initial_deposit: float = 1.0,
    ) -> None:
        super().__init__(unique_id, model)
        self.impatient: bool = impatient

        # Economic state
        self.deposit_t0: float = initial_deposit
        self.withdraw1: float = 0.0
        self.withdraw2: float = 0.0
        self.r1: float = 1.2
        self.R: float = 2.0
        self.fitness1: float = 0.0
        self.fitness2: float = 0.0

        # Bookkeeping
        self.active: bool = True
        self.served: bool = False

    # ---------------- NetLogo helpers, translated --------------------

    def _set_rates(self) -> None:
        """
        NetLogo 'set-rates' procedure. :contentReference[oaicite:4]{index=4}

        If 'fixed'   -> R = 2, r1 = 1.2
        If 'random'  -> R ~ U(1.2, 2.0), r1 ~ U(1.0, 1.2)
        """
        if self.model.rates_mode == "fixed":
            self.R = 2.0
            self.r1 = 1.2
        else:
            # Heterogeneous returns across agents and over time
            self.R = 1.2 + self.random.random() * 0.8
            self.r1 = 1.0 + self.random.random() * 0.2

    # ---------------- Decision rule (do-business-green) --------------

    def _decide_withdrawals(self) -> None:
        """
        Version of 'do-business-green' + part of 'fitness-check' 

        The logic is:

        - Impatient depositors:
            - Under 'constant' consumption:
                Withdraw a deterministic amount (1 if queuej > n_impatient,
                otherwise 1 + ε).
            - Under 'variable' consumption:
                Same base fitness, but the amount withdrawn is a random
                fraction w_j in (0,1).

        - Patient depositors:
            - Compute a "late" payoff V2 from the DD-style formula that
              depends on the fraction already served (queuej / N). :contentReference[oaicite:6]{index=6}
            - Compute the number of neighbors that have already withdrawn
              early (withdraw1 > 0).
            - If at least 3 neighbors have withdrawn early, the depositor
              joins the run and withdraws early; otherwise waits and takes
              the late payoff.
        """
        if not self.active or self.served:
            return

        # --- local NetLogo-like variables -----------------------------
        queuej = self.model.n_served  # n-served reporter
        n_impatient = self.model.num_impatients
        N = self.model.num_depositors
        consume1 = 1.0 + self.random.random() * 0.2  # 1 + random-float 0.2 :contentReference[oaicite:7]{index=7}
        n_withdraw1 = self.model.num_withdraw1

        # Reset withdrawals & fitness for this tick
        self.withdraw1 = 0.0
        self.withdraw2 = 0.0
        self.fitness1 = 0.0
        self.fitness2 = 0.0

        # Convenience: fraction already served
        frac_served = queuej / float(N) if N > 0 else 0.0

        # Late payoff (V2) following the active NetLogo formula:
        #   fitness2 = (R * (1 - (consume1 * (queuej / 441)))) / (1 - (queuej / 441))
        # guarding against division by zero when everyone has been served. 
        V2 = 0.0
        if frac_served < 1.0:
            V2 = self.R * (1.0 - (consume1 * frac_served)) / (1.0 - frac_served)

        if self.model.consumption_mode == "constant":
            # ----------------- CONSTANT CONSUMPTION -------------------
            if self.impatient:
                # NetLogo: if queuej > n-impatient set fitness1 1, else consume1
                if queuej > n_impatient:
                    self.fitness1 = 1.0
                else:
                    self.fitness1 = consume1
                self.withdraw1 = self.fitness1
            else:
                # Patient depositor with social network rule:
                # "depositors go to the bank if at least three of their proximate
                # neighbors went previously" 
                neighbors = self.model.grid.get_neighbors(
                    self.pos, moore=True, include_center=False
                )
                num_neighbors_withdraw1 = sum(
                    isinstance(n, DepositorAgent) and n.withdraw1 > 0 for n in neighbors
                )

                if num_neighbors_withdraw1 >= 3:
                    # Join the run: behave like an impatient agent.
                    self.fitness1 = consume1
                    self.withdraw1 = self.fitness1
                else:
                    # Wait and take the late payoff.
                    self.fitness2 = V2
                    self.withdraw2 = self.fitness2

        else:
            # ----------------- VARIABLE CONSUMPTION -------------------
            # Here we follow the same structure but allow a random fraction
            # w_j in (0,1) to scale the amount actually withdrawn. This
            # captures heterogeneous consumption schedules as described in
            # the paper. 
            w = self.random.random()

            if self.impatient:
                if queuej > n_impatient:
                    self.fitness1 = 1.0
                else:
                    self.fitness1 = consume1
                self.withdraw1 = w * self.fitness1
            else:
                neighbors = self.model.grid.get_neighbors(
                    self.pos, moore=True, include_center=False
                )
                num_neighbors_withdraw1 = sum(
                    isinstance(n, DepositorAgent) and n.withdraw1 > 0 for n in neighbors
                )

                if num_neighbors_withdraw1 >= 3:
                    # Join the run: early withdrawal with random share.
                    self.fitness1 = consume1
                    self.withdraw1 = w * self.fitness1
                else:
                    # Wait: draw a random share of the late payoff.
                    self.fitness2 = V2
                    self.withdraw2 = w * self.fitness2

        # Update deposits as in 'fitness-check' (we ignore account1/account2
        # beyond tracking remaining deposit, since the NetLogo plots only use
        # withdrawals and the bank's fin-balance). 
        total_withdrawn = self.withdraw1 + self.withdraw2
        self.deposit_t0 = max(self.deposit_t0 - total_withdrawn, 0.0)

        # Deactivate depositor if their deposit has become non-positive
        if self.deposit_t0 <= 0.0:
            self.active = False

    # ---------------- Public interface used by the Model --------------

    def serve(self) -> None:
        """
        Serve this depositor once in the sequential queue.

        This corresponds to one step in the 'go' procedure for this agent.
        """
        if not self.active or self.served:
            self.served = True
            return

        # Patch-level procedures: set-rates, do-business-green, fitness-check.
        self._set_rates()
        self._decide_withdrawals()
        self.served = True


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


class OneBankDDModel(Model):
    """
    One-bank Diamond–Dybvig ABM with social network effects.

    This is a Mesa version of DDmodelV4.nlogo as used in Romero (2009). 

    Parameters
    ----------
    width, height : int
        Grid dimensions. Default 21 x 21 -> 441 depositors (patches). :contentReference[oaicite:13]{index=13}
    impatient_probability : float
        Probability that a depositor is impatient (pcolor = green).
    rates_mode : {"fixed", "random"}
        Matches NetLogo GUI chooser 'rates'.
    consumption_mode : {"constant", "variable"}
        Matches NetLogo GUI chooser 'consumption'.
    initial_deposit : float
        Initial deposit per depositor (Table 1 uses 1).
    seed : int or None
        Optional random seed for reproducibility.
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

        # Parameters
        self.width = width
        self.height = height
        self.impatient_probability = impatient_probability
        self.rates_mode = rates_mode
        self.consumption_mode = consumption_mode
        self.initial_deposit = initial_deposit

        # Space & scheduler
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = BaseScheduler(self)

        # Bank at the center (same as NetLogo setxy 0 0 in a 21x21 torus) 
        self.bank = BankAgent("bank", self)
        self.schedule.add(self.bank)
        center = (width // 2, height // 2)
        self.grid.place_agent(self.bank, center)

        # Create depositors: one per patch
        self.depositors: List[DepositorAgent] = []
        uid = 0
        for x in range(width):
            for y in range(height):
                impatient = self.random.random() <= self.impatient_probability
                depositor = DepositorAgent(
                    unique_id=uid,
                    model=self,
                    impatient=impatient,
                    initial_deposit=self.initial_deposit,
                )
                self.depositors.append(depositor)
                self.schedule.add(depositor)
                self.grid.place_agent(depositor, (x, y))
                uid += 1

        self.num_depositors: int = len(self.depositors)

        # Initialize bank balance
        self.bank.init_deposits = sum(d.deposit_t0 for d in self.depositors)
        self.bank.fin_balance = self.bank.init_deposits
        self.bank.served = 0

        # Sequential service queue: random order of depositors
        self._service_queue: List[DepositorAgent] = self.random.sample(
            self.depositors, k=self.num_depositors
        )
        self._queue_index: int = 0

        # Data collection mirrors NetLogo monitors and plot "Accounts"
        self.datacollector = DataCollector(
            model_reporters={
                # Plot "Accounts"
                "deposits": lambda m: m.bank.fin_balance,
                "totalw": lambda m: m.total_withdrawals,
                "type1w": lambda m: m.total_withdraw1,
                "type2w": lambda m: m.total_withdraw2,
                # Monitors
                "withdrew": lambda m: m.total_withdrawals,
                "patients": lambda m: m.num_patients,
                "impatients": lambda m: m.num_impatients,
                "served": lambda m: m.n_served,
                # Bank status
                "bank_failed": lambda m: int(m.bank.fin_balance < 0.0),
            }
        )

        # Collect initial state at "tick 0"
        self.datacollector.collect(self)
        self.running = True

    # ---------------- Derived quantities (NetLogo reporters) ----------

    @property
    def num_impatients(self) -> int:
        """Equivalent of NetLogo reporter n-impatient."""
        return sum(1 for d in self.depositors if d.impatient)

    @property
    def num_patients(self) -> int:
        """Equivalent of NetLogo reporter n-patient."""
        return self.num_depositors - self.num_impatients

    @property
    def n_served(self) -> int:
        """Equivalent of NetLogo reporter n-served."""
        return sum(
            1
            for d in self.depositors
            if (d.withdraw1 > 0.0 or d.withdraw2 > 0.0)
        )

    @property
    def total_withdrawals(self) -> float:
        """Equivalent of NetLogo reporter t-served (sum of withdraw1+withdraw2)."""
        return sum(d.withdraw1 + d.withdraw2 for d in self.depositors)

    @property
    def total_withdraw1(self) -> float:
        return sum(d.withdraw1 for d in self.depositors)

    @property
    def total_withdraw2(self) -> float:
        return sum(d.withdraw2 for d in self.depositors)

    @property
    def num_withdraw1(self) -> int:
        """Count of agents with withdraw1 > 0, used inside do-business-green."""
        return sum(1 for d in self.depositors if d.withdraw1 > 0.0)

    # ---------------- Bank update (bank-balance-sheet) ----------------

    def _update_bank_balance(self) -> None:
        """
        Translation of 'bank-balance-sheet' with invest turned off. 

        fin-balance = init-deposits - total_withdrawals
        served      = n-served
        """
        tot_withdrawals = self.total_withdrawals
        self.bank.fin_balance = self.bank.init_deposits - tot_withdrawals
        self.bank.served = self.n_served

    # ---------------- Main time evolution -----------------------------

    def step(self) -> None:
        """
        One tick of the model.

        Each step corresponds to one depositor being served in a sequential
        queue, matching the way n-served and the Accounts plot evolve over
        time in the NetLogo implementation. 
        """
        if self._queue_index >= len(self._service_queue):
            # Everyone has been served
            self.running = False
            return

        # Serve next depositor in the queue
        agent = self._service_queue[self._queue_index]
        self._queue_index += 1
        agent.serve()

        # Update bank and collect data
        self._update_bank_balance()
        self.datacollector.collect(self)

        # If bank is insolvent, we stop as in NetLogo (bank turns red)
        if self.bank.fin_balance < 0.0:
            self.running = False

    def run_model(self, max_steps: Optional[int] = None) -> None:
        """
        Convenience method to run until everyone has been served or the
        bank fails, with an optional hard cap on steps.
        """
        steps = 0
        while self.running and (max_steps is None or steps < max_steps):
            self.step()
            steps += 1


__all__ = ["OneBankDDModel", "DepositorAgent", "BankAgent"]

