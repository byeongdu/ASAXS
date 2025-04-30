import xraydb
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox, QFileDialog, QWidget, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
# To install PyQt5, use the following command in your terminal or command prompt:
# pip install PyQt5
def recommend_input_energy(elements, include=None, exclude=None, margin_percent=20, min_energy_keV = 2, max_energy_keV=35, plot=True, save_plot=False, filename="recommended_energy_plot.png", save_pdf=False, pdf_filename="xrf_report.pdf"):
    """
    Recommend input X-ray energy based on specified include/exclude lists, with plotting and PDF report.
    
    Parameters:
        elements (list of str): List of all elements available in the sample.
        include (list of str): Elements to include in the energy recommendation.
        exclude (list of str): Elements to exclude from the energy recommendation.
        margin_percent (float): Margin above absorption edge.
        max_energy_keV (float): Maximum allowed energy.
        plot (bool): Show plot.
        save_plot (bool): Save plot as image.
        filename (str): Plot filename.
        save_pdf (bool): Save a PDF report.
        pdf_filename (str): PDF filename.
    """
    
    # Default to including all elements, if include list is not provided
    if include is None:
        include = elements
    
    # Default to excluding no elements, if exclude list is not provided
    if exclude is None:
        exclude = []
    
    edge_info = {}
    emission_lines_info = {}
    candidate_edges = []

    # Find K-edge or L3-edge for the included elements
    for elem in include:
        if elem in exclude:
            continue  # Skip the element if it is in the exclude list

        # Find K-edge or L3-edge
        k_edge = xraydb.xray_edge(elem, 'K')
        l3_edge = xraydb.xray_edge(elem, 'L3')

        chosen_edge = None
        edge_type = None

        if k_edge and (k_edge.energy/1000 <= max_energy_keV):
            chosen_edge = k_edge.energy
            edge_type = 'K'
        elif l3_edge:
            chosen_edge = l3_edge.energy
            edge_type = 'L3'
        else:
            print(f"Warning: No usable K or L3 edge found for {elem}")
        if chosen_edge:
            edge_info[elem] = (edge_type, chosen_edge)
            candidate_edges.append(chosen_edge)

    if not candidate_edges:
        raise ValueError("No suitable edges found under the maximum allowed energy.")

    #print(chosen_edge)
    highest_edge = max(candidate_edges)
    recommended_energy = highest_edge * (1 + margin_percent/100)
    print(f"Recommended energy (with margin): {recommended_energy/1000:.2f} keV and (highest edge: {highest_edge/1000:.2f} keV)")

    if recommended_energy/1000 > max_energy_keV:
        recommended_energy = max_energy_keV * 1000

    # Fetch emission lines for all elements (both include and exclude) and intensity
    for elem in elements:
        lines = xraydb.xray_lines(elem)
        emission_lines_info[elem] = []
        for line_name, line in lines.items():
            # Include K and L lines only
            if line.energy <= recommended_energy and line.energy > min_energy_keV*1000:  # Exclude lines below 2 keV
                print  (f"Element: {elem}, Line: {line_name}, Energy: {line.energy/1000:.2f} keV")
                emission_lines_info[elem].append((line_name, line.energy, line.intensity))

    # Plotting
    if plot or save_plot:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot edges
        element_colors = plt.cm.tab20.colors  # Use a colormap for distinct colors
        element_color_map = {elem: element_colors[i % len(element_colors)] for i, elem in enumerate(elements)}

        marker_style = 'o'  # Closed circle for absorption edges
        for elem, (etype, evalue) in edge_info.items():
            marker_color = element_color_map[elem]
            ax.plot(evalue/1000, 0, marker_style, color=marker_color, label=f"{elem} {etype}-edge ({evalue/1000:.2f} keV)")

        # Plot emission lines
        marker_style = 'x'  # 'x' for emission lines
        for elem, lines in emission_lines_info.items():
            for (label, energy, intensity) in lines:
                marker_color = element_color_map[elem]
                ax.plot(energy/1000, intensity, marker_style, color=marker_color)
                ax.text(energy/1000, intensity, f"{elem} {label}", rotation=90, fontsize=8, ha='center', va='top')
        # for elem, (etype, evalue) in edge_info.items():
        #     marker_color = 'blue' if etype == 'K' else 'green'
        #     ax.plot(evalue/1000, 0, 'o', color=marker_color, label=f"{elem} {etype}-edge ({evalue/1000:.2f} keV)")
        
        # # Plot emission lines
        # for elem, lines in emission_lines_info.items():
        #     for (label, energy, intensity) in lines:
        #         ax.plot(energy/1000, intensity, 'x', color='orange')
        #         ax.text(energy/1000, -0.2, f"{elem} {label}", rotation=90, fontsize=8, ha='center', va='top')
        
        ax.axvline(recommended_energy/1000, color='red', linestyle='--', label=f"Recommended Energy ({recommended_energy/1000:.2f} keV)")

        ax.set_xlabel('Energy (keV)')
        ax.set_yticks([])
        ax.set_title('Absorption Edges, Emission Lines, and Recommended Input Energy')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(loc='best', fontsize=8)
        plt.tight_layout()

        if save_plot:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {filename}")
        
        if plot:
            plt.show()
        else:
            plt.close()

    # Create PDF
    if save_pdf:
        c = canvas.Canvas(pdf_filename, pagesize=letter)
        width, height = letter
        
        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(1 * inch, height - 1 * inch, "XRF Input Energy Report")
        
        # Recommended Energy
        c.setFont("Helvetica", 12)
        c.drawString(1 * inch, height - 1.5 * inch, f"Recommended Input Energy: {recommended_energy/1000:.2f} keV")
        
        # Edges
        c.drawString(1 * inch, height - 2.0 * inch, "Element Edges:")
        y = height - 2.3 * inch
        for elem, (etype, evalue) in edge_info.items():
            c.drawString(1.2 * inch, y, f"{elem}: {etype}-edge at {evalue/1000:.2f} keV")
            y -= 0.2 * inch
        
        # Emission lines and intensities
        c.drawString(1 * inch, y, "Emission Lines and Intensities:")
        y -= 0.3 * inch
        for elem, lines in emission_lines_info.items():
            for (line, energy, intensity) in lines:
                c.drawString(1.2 * inch, y, f"{elem}: {line} at {energy/1000:.2f} keV, Intensity: {intensity:.2f}")
                y -= 0.2 * inch
        
        # Plot image
        if save_plot and os.path.exists(filename):
            c.drawImage(filename, 1 * inch, 0.5 * inch, width=5.5*inch, preserveAspectRatio=True, mask='auto')
        
        c.save()
        print(f"PDF report saved as: {pdf_filename}")

    return recommended_energy, edge_info, emission_lines_info

class XRFInputGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XRF Input Energy Configuration")
        self.setGeometry(100, 100, 1200, 800)

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # Elements input
        self.elements_label = QLabel("Elements (comma-separated):")
        self.elements_input = QLineEdit()
        layout.addWidget(self.elements_label)
        layout.addWidget(self.elements_input)

        # Exclude elements input
        self.exclude_label = QLabel("Exclude Elements (comma-separated):")
        self.exclude_input = QLineEdit()
        layout.addWidget(self.exclude_label)
        layout.addWidget(self.exclude_input)

        # Low energy limit input
        self.low_energy_label = QLabel("Low Energy Limit for Emission Output (keV, default 2):")
        self.low_energy_input = QLineEdit()
        self.low_energy_input.setPlaceholderText("2")
        layout.addWidget(self.low_energy_label)
        layout.addWidget(self.low_energy_input)

        # Save plot checkbox
        self.save_plot_checkbox = QCheckBox("Save Plot")
        layout.addWidget(self.save_plot_checkbox)

        # Save PDF checkbox
        self.save_pdf_checkbox = QCheckBox("Save PDF")
        layout.addWidget(self.save_pdf_checkbox)
        # Plot canvas
        self.plot_canvas = FigureCanvas(Figure(figsize=(12, 10)))
        layout.addWidget(self.plot_canvas)

        # Run button
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_recommendation)
        layout.addWidget(self.run_button)
        central_widget.setLayout(layout)

    def run_recommendation(self):
        try:
            # Get user inputs
            elements = [e.strip() for e in self.elements_input.text().split(",") if e.strip()]
            exclude_elements = [e.strip() for e in self.exclude_input.text().split(",") if e.strip()]
            include_elements = [e for e in elements if e not in exclude_elements]
            print(f"Include elements: {include_elements}")
            low_energy_limit = float(self.low_energy_input.text()) if self.low_energy_input.text().strip() else 2.0
            save_plot = self.save_plot_checkbox.isChecked()
            save_pdf = self.save_pdf_checkbox.isChecked()
            # Run the function without plotting
            recommend_energy, edges, emissions = recommend_input_energy(
                elements,
                include=include_elements,
                exclude=exclude_elements,
                margin_percent=20.0,
                min_energy_keV=low_energy_limit,
                max_energy_keV=35,
                plot=False,  # Disable plotting in the function
                save_plot=save_plot,
                filename="xrf_recommended_energy_with_emissions_and_intensity.png",
                save_pdf=save_pdf,
                pdf_filename="xrf_report_with_emissions_and_intensity.pdf"
            )

            # Embed the plot in the GUI
            self.plot_canvas.figure.clear()
            ax = self.plot_canvas.figure.add_subplot(111)
            element_colors = plt.cm.tab20.colors
            element_color_map = {elem: element_colors[i % len(element_colors)] for i, elem in enumerate(elements)}

            # Plot edges
            for elem, (etype, evalue) in edges.items():
                marker_color = element_color_map[elem]
                ax.plot(evalue / 1000, 0, 'o', color=marker_color, label=f"{elem} {etype}-edge ({evalue / 1000:.2f} keV)")

            # Plot emission lines
            for elem, lines in emissions.items():
                for (label, energy, intensity) in lines:
                    marker_color = element_color_map[elem]
                    ax.plot(energy / 1000, intensity, 'x', color=marker_color)
                    ax.text(energy / 1000, intensity, f"{elem} {label}", rotation=90, fontsize=12, ha='center', va='top')

            ax.axvline(recommend_energy / 1000, color='red', linestyle='--', label=f"Recommended Energy ({recommend_energy / 1000:.2f} keV)")
            ax.set_xlabel('Energy (keV)')
            ax.set_yticks([])
            ax.set_title('Absorption Edges, Emission Lines, and Recommended Input Energy')
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            ax.legend(loc='best', fontsize=8)
            self.plot_canvas.draw()
            #     save_plot=save_plot,
            #     filename="xrf_recommended_energy_with_emissions_and_intensity.png",
            #     save_pdf=save_pdf,
            #     pdf_filename="xrf_report_with_emissions_and_intensity.pdf"
            # )

            # Show results
            result_message = f"Recommended input energy: {energy/1000:.2f} keV\n\nEdge energies used:\n"
            for elem, (etype, evalue) in edges.items():
                result_message += f"  {elem}: {etype}-edge at {evalue/1000:.2f} keV\n"

            result_message += "\nEmission lines and intensities:\n"
            for elem, lines in emissions.items():
                for (line, energy, intensity) in lines:
                    result_message += f"  {elem}: {line} at {energy/1000:.2f} keV, Intensity: {intensity:.2f}\n"

            QMessageBox.information(self, "Results", result_message)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = XRFInputGUI()
    gui.show()
    sys.exit(app.exec_())

# energy, edges, emissions = recommend_input_energy(
#     elements,
#     include=include_elements,
#     exclude=exclude_elements,
#     save_plot=True,
#     filename="xrf_recommended_energy_with_emissions_and_intensity.png",
#     save_pdf=True,
#     pdf_filename="xrf_report_with_emissions_and_intensity.pdf"
# )

# print(f"\nRecommended input energy: {energy/1000:.2f} keV")
# print("Edge energies used:")
# for elem, (etype, evalue) in edges.items():
#     print(f"  {elem}: {etype}-edge at {evalue/1000:.2f} keV")

# print("\nEmission lines and intensities:")
# for elem, lines in emissions.items():
#     for (line, energy, intensity) in lines:
#         print(f"  {elem}: {line} at {energy/1000:.2f} keV, Intensity: {intensity:.2f}")
