from dataclasses import dataclass
from pprint import pp
import sys

@dataclass
class Architecture:
    name: str
    cost: float
    power: float
    time_sudoku: float
    time_conv: float
    time_inference: float

    @property
    def energy_sudoku(self):
        return self.time_sudoku / 1000 * self.power

    @property
    def energy_conv(self):
        return self.time_conv / 1000 * self.power

    @property
    def energy_inference(self):
        return self.time_inference / 1000 * self.power

    @property
    def edp_sudoku(self):
        return self.time_sudoku * self.energy_sudoku

    @property
    def edp_conv(self):
        return self.time_conv * self.energy_conv

    @property
    def edp_inference(self):
        return self.time_inference * self.energy_inference
    
    @property
    def cost_performance_sudoku(self):
        return self.time_sudoku / 1000 * self.cost

    @property
    def cost_performance_conv(self):
        return self.time_conv / 1000* self.cost

    @property
    def cost_performance_inference(self):
        return self.time_inference / 1000 * self.cost

    @property
    def cost_energy_sudoku(self):
        return self.energy_sudoku * self.cost

    @property
    def cost_energy_conv(self):
        return self.energy_conv * self.cost

    @property
    def cost_energy_inference(self):
        return self.energy_inference * self.cost

CPU = Architecture("CPU", cost=140, power=65, time_sudoku=0.5318, time_conv=608.3400, time_inference=304.4000)
GPU = Architecture("GPU", cost=400, power=160, time_sudoku=11.1800, time_conv=0.8732, time_inference=151.5980)
NPU = Architecture("NPU", cost=160, power=5, time_sudoku=10.7130, time_conv=225.0000, time_inference=6.6850)

arqs = [CPU, GPU, NPU]
valid_metrics = {
    "cost_performance",
    "cost_energy",
    "edp",
    "energy",
    "time",
}


def generate_design_space():
    design_space = []
    for sudoku_opt in arqs:
        for conv_opt in arqs:
            for inference_opt in arqs:
                total_time = sudoku_opt.time_sudoku + conv_opt.time_conv + inference_opt.time_inference
                total_energy = sudoku_opt.energy_sudoku + conv_opt.energy_conv + inference_opt.energy_inference
                total_edp = sudoku_opt.edp_sudoku + conv_opt.edp_conv + inference_opt.edp_inference
                total_cost_performance = sudoku_opt.cost_performance_sudoku + conv_opt.cost_performance_conv + inference_opt.cost_performance_inference
                total_cost_energy = sudoku_opt.cost_energy_sudoku + conv_opt.cost_energy_conv + inference_opt.cost_energy_inference

                design_space.append((
                    {
                        "cost_performance": total_cost_performance, 
                        "cost_energy": total_cost_energy, 
                        "edp": total_edp, 
                        "energy": total_energy, 
                        "time": total_time
                    }, 
                    (sudoku_opt, conv_opt, inference_opt)
                ))
    return design_space

def rank_alternatives(design_space, metric):
    return sorted(design_space, key=lambda item: item[0][metric])



def main():
    if len(sys.argv) != 2 or sys.argv[1] not in valid_metrics:
        print("Provide the desired metric to rank the alternatives")
        print(f"Valid metrics: {valid_metrics}")
        return

    metric = sys.argv[1]

    design_space = generate_design_space()
    ranking = rank_alternatives(design_space, metric)

    for rank, settings in enumerate(ranking, start=1):
        metrics, allocation = settings
        print(f"{rank}) {allocation[0].name} {allocation[1].name} {allocation[2].name}")
        pp(metrics)

if __name__ == '__main__':
    main()