# === core/strategy_base.py ===

class Strategy:
    def configure(self, params):
        # Default configure method to avoid frozen config errors
        for k, v in params.items():
            setattr(self, k, v)

    def run(self):
        raise NotImplementedError("Strategy must implement run method.")

    def log_results(self):
        pass
