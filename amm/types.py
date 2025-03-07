class TokenPair:
    def __init__(self, token0: str, token1: str, isEthereum: bool = False):
        self.token0 = "ETH" if isEthereum and token0.upper() == "WETH" else token0
        self.token1 = "ETH" if isEthereum and token1.upper() == "WETH" else token1

    def __str__(self):
        return f"{self.token0.upper()}-{self.token1.upper()}" if self.token0 < self.token1 else f"{self.token1.upper()}-{self.token0.upper()}"