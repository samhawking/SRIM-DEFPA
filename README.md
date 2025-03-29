# SRIM-DEFPA

SRIM-DEFPA is a Python-based GUI tool designed to analyze radiation effects on materials by calculating **defect concentrations** and **displacements per atom (DPA)** in semiconductor layers using data from **SRIM** simulations. It processes SRIM's `VACANCY.txt` files, allowing users to input parameters such as layer thicknesses and proton fluence to compute vacancy densities and defect concentrations across multiple layers with high accuracy, thanks to interpolation and numerical integration techniques. This ensures precise defect mapping, making it an ideal tool for applications in **solar cells**, **semiconductors**, and **aerospace materials**. With its user-friendly interface, SRIM-DEFPA enables researchers to visualize radiation damage through bar charts and save plots for further analysis, providing valuable insights into radiation-induced degradation and supporting the design of radiation-hardened devices.

## Features
- **User-friendly GUI** for inputting material properties and radiation parameters.
- **Automated data processing** from SRIM output files.
- **Precise defect concentration calculations** for various materials.
- **DPA computation** based on radiation fluence and energy.
- **Visualization tools** for analyzing simulation results.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/samhawking/SRIM-DEFPA.git
   ```
2. Navigate to the project directory:
   ```bash
   cd SRIM-DEFPA
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python srim_defpa.py
   ```

## Usage
- Open the tool and input **layer thickness, energy, and fluence**.
- The program selects appropriate **depth values** from SRIM data.
- Computes **defects concentration** and **DPA**.
- View results and export data for further analysis.

## Contribution
Feel free to contribute by submitting issues or pull requests.

## Contact
- **GitHub Profile**: [samhawking](https://github.com/samhawking)
- **Email**: phyzaoui@gmail.com

---
### License
This project is licensed under the MIT License.

