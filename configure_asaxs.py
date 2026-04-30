import sys
import numpy as np
import pyqtgraph as pg
import xraydb

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout, QFileDialog
from PyQt5.QtCore import Qt

pg.setConfigOptions(background='w', foreground='k')

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

        self.mono_offset_label = QLabel("Mono Offset (keV):")
        input_layout.addWidget(self.mono_offset_label)
        self.mono_offset_input = QLineEdit()
        self.mono_offset_input.setText("0.0")
        input_layout.addWidget(self.mono_offset_input)

        self.layout.addLayout(input_layout)

        self.start_button = QPushButton("Start Analysis")
        self.start_button.clicked.connect(self.start_analysis)
        self.layout.addWidget(self.start_button)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.layout.addWidget(self.result_text)

        save_layout = QHBoxLayout()
        self.order_button = QPushButton("Order: Increasing ↑")
        self.order_button.setCheckable(True)
        self.order_button.toggled.connect(self._toggle_order)
        save_layout.addWidget(self.order_button)
        self.save_button = QPushButton("Save Results to TXT")
        self.save_button.clicked.connect(self.save_results)
        save_layout.addWidget(self.save_button)
        self.layout.addLayout(save_layout)

        # Message log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(80)
        self.layout.addWidget(self.log_text)

        # pyqtgraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)

        # Data containers
        self.selected_energies = []
        self.selected_f1 = []
        self.element = ""
        self.edge_info = ""
        self._vb2 = None
        self._update_views = None

    def _log(self, msg):
        self.log_text.append(msg)

    def get_absorption_edge(self, element, min_energy_keV=6, max_energy_keV=34):
        k_edge = xraydb.xray_edge(element, 'K')
        l3_edge = xraydb.xray_edge(element, 'L3')

        chosen_edge = None
        edge_type = None
        self._log(f"Element: {element}, K-edge: {k_edge}, L3-edge: {l3_edge}")
        if k_edge and (min_energy_keV <= k_edge.energy/1000 <= max_energy_keV):
            chosen_edge = k_edge.energy
            edge_type = 'K'
        elif l3_edge and (min_energy_keV <= l3_edge.energy/1000 <= max_energy_keV):
            chosen_edge = l3_edge.energy
            edge_type = 'L3'

        return chosen_edge, edge_type

    def get_f1_curve(self, element, edge_energy_eV, energy_range_eV=200, num_points=300, above_eV=100):
        energies = np.linspace(edge_energy_eV - energy_range_eV, edge_energy_eV + above_eV, num_points)
        f1_values = np.array([xraydb.f1_chantler(element, en) for en in energies])
        f2_values = np.array([xraydb.f2_chantler(element, en) for en in energies])
        return energies, f1_values, f2_values

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
        self.element = self.element_input.text().strip().capitalize()

        if not self.element:
            self.result_text.setText("Please enter a valid element symbol.")
            return

        try:
            num_points = int(self.num_points_input.text())
            energy_range = float(self.energy_range_input.text())
            mono_offset = float(self.mono_offset_input.text())
        except ValueError:
            self.result_text.setText("Please enter valid numbers for Number of Points, Energy Range, and Mono Offset.")
            return

        if num_points < 2:
            self.result_text.setText("Number of Points should be at least 2.")
            return

        edge_energy, edge_type = self.get_absorption_edge(self.element)

        if edge_energy is None:
            self.result_text.setText(f"No suitable K or L3 edge found for {self.element} between 6 and 34 keV.")
            return
        
        self.edge_info = f"Using {edge_type}-edge at {edge_energy/1000:.2f} keV for {self.element}."

        energies, f1_values, f2_values = self.get_f1_curve(self.element, edge_energy, energy_range)
        mask = energies <= edge_energy
        self.selected_energies, self.selected_f1 = self.select_uniform_f1_points(
            energies[mask], f1_values[mask], num_points)
        self.selected_energies = self.selected_energies + mono_offset * 1000

        self._update_result_text()

        # Clean up previous secondary ViewBox before clearing the plot
        if self._vb2 is not None:
            try:
                self.plot_widget.getPlotItem().vb.sigResized.disconnect(self._update_views)
            except Exception:
                pass
            self.plot_widget.scene().removeItem(self._vb2)
            self._vb2 = None

        # Plot
        self.plot_widget.clear()
        self.plot_widget.setLabel('bottom', "Energy (keV)")
        self.plot_widget.setLabel('left', "f1", color='b')
        self.plot_widget.setTitle(f"{self.element} - f1 and f2 near {edge_type}-edge")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()
        self.plot_widget.plot(energies/1000, f1_values, pen=pg.mkPen('b', width=2), name="f1 curve")
        self.plot_widget.plot(self.selected_energies/1000, self.selected_f1,
                              pen=None, symbol='o', symbolBrush='r', symbolSize=8,
                              name="Selected Points")

        # Secondary y-axis for f2
        plot_item = self.plot_widget.getPlotItem()
        self._vb2 = pg.ViewBox()
        plot_item.showAxis('right')
        plot_item.scene().addItem(self._vb2)
        plot_item.getAxis('right').linkToView(self._vb2)
        plot_item.getAxis('right').setLabel('f2', color='g')
        self._vb2.setXLink(plot_item)
        self._vb2.addItem(pg.PlotCurveItem(energies/1000, f2_values,
                                           pen=pg.mkPen('g', width=2)))
        # Dummy entry so f2 appears in the legend
        self.plot_widget.plot([], [], pen=pg.mkPen('g', width=2), name="f2 curve")

        def update_views():
            self._vb2.setGeometry(plot_item.vb.sceneBoundingRect())
            self._vb2.linkedViewChanged(plot_item.vb, self._vb2.XAxis)

        self._update_views = update_views
        update_views()
        plot_item.vb.sigResized.connect(update_views)

    def _update_result_text(self):
        if len(self.selected_energies) == 0:
            return
        decreasing = self.order_button.isChecked()
        energies_out = self.selected_energies[::-1] if decreasing else self.selected_energies
        f1_out = self.selected_f1[::-1] if decreasing else self.selected_f1
        output = f"{self.edge_info}\n\nSelected {len(energies_out)} energy points:\n"
        for en, f1 in zip(energies_out, f1_out):
            output += f"  Energy: {en:.1f} eV, f1: {f1:.3f}\n"
        self.result_text.setText(output)

    def _toggle_order(self, checked):
        self.order_button.setText("Order: Decreasing ↓" if checked else "Order: Increasing ↑")
        self._update_result_text()

    def save_results(self):
        if len(self.selected_energies) == 0:
            self.result_text.setText("No results to save. Please run the analysis first.")
            return

        decreasing = self.order_button.isChecked()
        energies_out = self.selected_energies[::-1] if decreasing else self.selected_energies
        f1_out = self.selected_f1[::-1] if decreasing else self.selected_f1

        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "Text Files (*.txt)", options=options)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(f"{self.edge_info}\n")
                f.write("Energy (eV)\tf1\n")
                for en, f1v in zip(energies_out, f1_out):
                    f.write(f"{en:.1f}\t{f1v:.3f}\n")
            self._log(f"Results saved to {filepath}")
        np.savetxt(f"{self.element}_selected_energies.txt", energies_out, fmt="%.3f")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ASAXSconfig()
    window.show()
    sys.exit(app.exec())
