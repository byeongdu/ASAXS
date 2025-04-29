import sys
import numpy as np
import matplotlib.pyplot as plt
import xraydb

from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout, QFileDialog
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ASAXSconfig(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASAXS - X-ray Absorption Edge Analysis")
        self.setGeometry(300, 100, 800, 850)
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Input boxes
        self.label = QLabel("Enter Element Symbol (e.g., Fe, Mo, Pb):")
        self.layout.addWidget(self.label)

        self.element_input = QLineEdit()
        self.layout.addWidget(self.element_input)

        input_layout = QHBoxLayout()
        
        self.num_points_label = QLabel("Number of Points:")
        input_layout.addWidget(self.num_points_label)
        self.num_points_input = QLineEdit()
        self.num_points_input.setText("15")
        input_layout.addWidget(self.num_points_input)

        self.energy_range_label = QLabel("Low Energy Range (eV):")
        input_layout.addWidget(self.energy_range_label)
        self.energy_range_input = QLineEdit()
        self.energy_range_input.setText("200")
        input_layout.addWidget(self.energy_range_input)

        self.layout.addLayout(input_layout)

        self.start_button = QPushButton("Start Analysis")
        self.start_button.clicked.connect(self.start_analysis)
        self.layout.addWidget(self.start_button)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.layout.addWidget(self.result_text)

        self.save_button = QPushButton("Save Results to TXT")
        self.save_button.clicked.connect(self.save_results)
        self.layout.addWidget(self.save_button)

        # Matplotlib figure
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Data containers
        self.selected_energies = []
        self.selected_f1 = []
        self.element = ""
        self.edge_info = ""

    def get_absorption_edge(self, element, min_energy_keV=6, max_energy_keV=34):
        k_edge = xraydb.xray_edge(element, 'K')
        l3_edge = xraydb.xray_edge(element, 'L3')

        chosen_edge = None
        edge_type = None
        print(f"Element: {element}, K-edge: {k_edge}, L3-edge: {l3_edge}")
        if k_edge and (min_energy_keV <= k_edge.energy/1000 <= max_energy_keV):
            chosen_edge = k_edge.energy
            edge_type = 'K'
        elif l3_edge and (min_energy_keV <= l3_edge.energy/1000 <= max_energy_keV):
            chosen_edge = l3_edge.energy
            edge_type = 'L3'

        return chosen_edge, edge_type

    def get_f1_curve(self, element, edge_energy_eV, energy_range_eV=200, num_points=300):
        energies = np.linspace(edge_energy_eV - energy_range_eV, edge_energy_eV, num_points)
        f1_values = np.array([xraydb.f1_chantler(element, en) for en in energies])
        return energies, f1_values

    def select_uniform_f1_points(self, energies, f1_values, num_points=15):
        sorted_idx = np.argsort(f1_values)
        f1_sorted = f1_values[sorted_idx]
        energy_sorted = energies[sorted_idx]

        f1_min, f1_max = f1_sorted[0], f1_sorted[-1]
        f1_targets = np.linspace(f1_min, f1_max, num_points)

        selected_energies = []
        selected_f1 = []

        for target in f1_targets:
            idx = np.abs(f1_sorted - target).argmin()
            selected_energies.append(energy_sorted[idx])
            selected_f1.append(f1_sorted[idx])

        # sort back by energy
        selected = sorted(zip(selected_energies, selected_f1), key=lambda x: x[0])
        energies_out, f1_out = zip(*selected)

        return np.array(energies_out), np.array(f1_out)

    def start_analysis(self):
        self.result_text.clear()
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        self.element = self.element_input.text().strip().capitalize()

        if not self.element:
            self.result_text.setText("Please enter a valid element symbol.")
            return

        try:
            num_points = int(self.num_points_input.text())
            energy_range = float(self.energy_range_input.text())
        except ValueError:
            self.result_text.setText("Please enter valid numbers for Number of Points and Energy Range.")
            return

        if num_points < 2:
            self.result_text.setText("Number of Points should be at least 2.")
            return

        edge_energy, edge_type = self.get_absorption_edge(self.element)

        if edge_energy is None:
            self.result_text.setText(f"No suitable K or L3 edge found for {self.element} between 6 and 34 keV.")
            return
        
        self.edge_info = f"Using {edge_type}-edge at {edge_energy/1000:.2f} keV for {self.element}."

        energies, f1_values = self.get_f1_curve(self.element, edge_energy, energy_range)
        self.selected_energies, self.selected_f1 = self.select_uniform_f1_points(energies, f1_values, num_points)

        # Update text output
        output = f"{self.edge_info}\n\nSelected {num_points} energy points:\n"
        for en, f1 in zip(self.selected_energies, self.selected_f1):
            output += f"  Energy: {en:.1f} eV, f1: {f1:.3f}\n"
        
        self.result_text.setText(output)

        # Plot
        ax.plot(energies/1000, f1_values, label=f"{self.element} f1 curve")
        ax.plot(self.selected_energies/1000, self.selected_f1, 'ro', label="Selected Points")
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("f1")
        ax.set_title(f"{self.element} - f1 curve near {edge_type}-edge")
        ax.grid(True)
        ax.legend()
        self.canvas.draw()

    def save_results(self):
        if not self.selected_energies:
            self.result_text.setText("No results to save. Please run the analysis first.")
            return

        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "Text Files (*.txt)", options=options)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(f"{self.edge_info}\n")
                f.write("Energy (eV)\tf1\n")
                for en, f1v in zip(self.selected_energies, self.selected_f1):
                    f.write(f"{en:.1f}\t{f1v:.3f}\n")
            self.result_text.append(f"\nResults saved to {filepath}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ASAXSconfig()
    window.show()
    sys.exit(app.exec())
