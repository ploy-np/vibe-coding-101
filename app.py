import statistics
import time
from typing import Dict, List, Optional, Tuple

# =====================================================================
# CONFIGURATION SWITCH
# =====================================================================
# Set to True to execute the 3-Tier Testing Suite. 
# Set to False to launch the interactive application.
RUN_TEST_SUITE = False

# Set to True to run the application in console.
# Set to True to run the application with GUI.
RUN_IN_CONSOLE = False

# =====================================================================
# 1. DATA ACCESS LAYER (MODEL)
# =====================================================================
class PatientModel:
    """Manages the raw patient data storage, retrieval, and initial data cleaning."""
    
    def __init__(self):
        # Initial mockup data containing one anomaly (BMI = 0 for Patient 102)
        self._raw_patients: Dict[int, Dict[str, float]] = {
            101: {"Glucose": 95.0, "BMI": 22.5, "Age": 28.0, "BloodPressure": 115.0},
            102: {"Glucose": 145.0, "BMI": 0.0, "Age": 54.0, "BloodPressure": 135.0}, # Anomaly
            103: {"Glucose": 112.0, "BMI": 29.1, "Age": 42.0, "BloodPressure": 122.0},
            104: {"Glucose": 180.0, "BMI": 36.4, "Age": 61.0, "BloodPressure": 142.0}
        }
        self._clean_initial_data()

    def _clean_initial_data(self) -> None:
        """Finds any BMI of 0 and replaces it with the median BMI of valid patients."""
        valid_bmis = [p["BMI"] for p in self._raw_patients.values() if p["BMI"] > 0]
        median_bmi = statistics.median(valid_bmis) if valid_bmis else 25.0
        
        for pid, metrics in self._raw_patients.items():
            if metrics["BMI"] <= 0:
                metrics["BMI"] = round(median_bmi, 1)

    def get_all_ids(self) -> List[int]:
        """Returns a list of all existing Patient IDs."""
        return sorted(list(self._raw_patients.keys()))

    def get_patient(self, patient_id: int) -> Optional[Dict[str, float]]:
        """Retrieves a deep copy of a specific patient's metrics."""
        patient = self._raw_patients.get(patient_id)
        return patient.copy() if patient else None

    def update_patient(self, patient_id: int, updated_metrics: Dict[str, float]) -> bool:
        """Updates an existing patient's clinical profile."""
        if patient_id in self._raw_patients:
            self._raw_patients[patient_id].update(updated_metrics)
            return True
        return False


# =====================================================================
# 2. BUSINESS LOGIC LAYER (SERVICE)
# =====================================================================
class ClinicalRiskService:
    """Handles clinical decision rules, point scoring, and risk categorization."""
    
    # Threshold rules: (Low max, Medium max). Anything above Medium max is High.
    THRESHOLDS = {
        "Glucose": (100.0, 125.0),
        "BMI": (25.0, 29.9),
        "Age": (35.0, 55.0),
        "BloodPressure": (120.0, 130.0)
    }

    def calculate_metric_score(self, metric_name: str, value: float) -> int:
        """Calculates point score (0, 1, or 2) for a specific clinical metric."""
        if metric_name not in self.THRESHOLDS:
            return 0
        low_max, med_max = self.THRESHOLDS[metric_name]
        if value <= low_max:
            return 0
        elif value <= med_max:
            return 1
        else:
            return 2

    def evaluate_patient_risk(self, metrics: Dict[str, float]) -> Tuple[int, str]:
        """Calculates aggregate score and maps it to a clinical risk category."""
        total_score = 0
        for metric, value in metrics.items():
            total_score += self.calculate_metric_score(metric, value)
            
        if total_score <= 2:
            category = "Low Risk"
        elif total_score <= 5:
            category = "Moderate Risk"
        else:
            category = "High Risk"
            
        return total_score, category


# =====================================================================
# 3. PRESENTATION LAYER (TEXT-BASED CONSOLE)
# =====================================================================
class ConsoleView:
    """Handles system layout, user text inputs, and structured reports."""
    
    @staticmethod
    def display_main_menu() -> str:
        print("\n" + "="*40)
        print("     DIABETES RISK SCORING SYSTEM")
        print("="*40)
        print("1. Assess Patient Risk")
        print("2. Exit")
        print("-"*40)
        return input("Select an option (1-2): ").strip()

    @staticmethod
    def display_patient_ids(ids: List[int]) -> None:
        print(f"\nAvailable Patient IDs: {', '.join(map(str, ids))}")

    @staticmethod
    def prompt_patient_id() -> str:
        return input("Enter Patient ID to assess: ").strip()

    @staticmethod
    def display_error(message: str) -> None:
        print(f"\n[ERROR] {message}")

    @staticmethod
    def display_profile(patient_id: int, metrics: Dict[str, float]) -> None:
        print(f"\n--- Clinical Profile for Patient {patient_id} ---")
        for metric, val in metrics.items():
            print(f" * {metric}: {val}")

    @staticmethod
    def prompt_modification_choice() -> bool:
        choice = input("\nDo you want to modify any metrics before calculation? (y/n): ").strip().lower()
        return choice == 'y'

    @staticmethod
    def prompt_metric_update(metric_name: str, current_value: float) -> float:
        user_input = input(f"Enter new {metric_name} [Current: {current_value}] (Or press Enter to keep): ").strip()
        if user_input == "":
            return current_value
        try:
            return float(user_input)
        except ValueError:
            print("[Invalid Input] Keeping original value.")
            return current_value

    @staticmethod
    def display_diagnostic_report(patient_id: int, score: int, category: str) -> None:
        print("\n" + "*"*40)
        print("          DIAGNOSTIC RISK REPORT")
        print("*"*40)
        print(f" Patient ID:       {patient_id}")
        print(f" Cumulative Score: {score} pts")
        print(f" Risk Category:    {category.upper()}")
        print("*"*40)

# =====================================================================
# 3. PRESENTATION LAYER (WEB-BASED STREAMLIT)
# =====================================================================

class StreamlitView:
    """Handles system layout, user interactions, and visual reports via a web-based UI."""
    
    @staticmethod
    def render_ui(model: PatientModel, service: ClinicalRiskService) -> None:
        import streamlit as st
        
        st.set_page_config(page_title="Diabetes Risk Scoring System", page_icon="🩺", layout="centered")
        
        st.title("🩺 Diabetes Risk Scoring System")
        st.markdown("---")
        
        # 1. Fetch available patients
        valid_ids = model.get_all_ids()
        
        # 2. Patient Selection Panel
        st.sidebar.header("Patient Selection")
        selected_id = st.sidebar.selectbox("Select Patient ID to Assess", options=valid_ids)
        
        if selected_id:
            # Fetch patient data deep copy
            patient_metrics = model.get_patient(selected_id)
            
            st.subheader(f"Clinical Profile: Patient {selected_id}")
            
            # 3. Input Fields / Form Panel for modifications
            with st.form(key=f"patient_form_{selected_id}"):
                updated_metrics = {}
                cols = st.columns(2)
                
                for idx, (metric, current_val) in enumerate(patient_metrics.items()):
                    col = cols[idx % 2]
                    updated_metrics[metric] = col.number_input(
                        label=f"{metric}",
                        value=float(current_val),
                        step=0.1,
                        format="%.1f"
                    )
                
                submit_button = st.form_submit_button(label="Evaluate Risk Assessment")
            
            # 4. Handle form submission and update database context safely
            if submit_button:
                model.update_patient(selected_id, updated_metrics)
                
                # 5. Evaluate and display report panel
                score, category = service.evaluate_patient_risk(updated_metrics)
                
                st.markdown("---")
                st.subheader("📊 Diagnostic Risk Report")
                
                # Match visual coloring container based on risk profile severity
                category_upper = category.upper()
                if "HIGH" in category_upper:
                    st.error(f"**Risk Category:** {category_upper}")
                elif "MODERATE" in category_upper:
                    st.warning(f"**Risk Category:** {category_upper}")
                else:
                    st.success(f"**Risk Category:** {category_upper}")
                
                col_metric1, col_metric2 = st.columns(2)
                col_metric1.metric(label="Patient ID", value=selected_id)
                col_metric2.metric(label="Cumulative Score", value=f"{score} pts")

# =====================================================================
# 4. ORCHESTRATION LAYER (CONTROLLER)
# =====================================================================
class RiskAssessmentController:
    """Coordinates interaction workflows between Model, Service, and View."""
    
    def __init__(self, model: PatientModel, service: ClinicalRiskService, view: ConsoleView):
        self.model = model
        self.service = service
        self.view = view

    def run(self) -> None:
        """Main operational runtime loop."""
        while True:
            choice = self.view.display_main_menu()
            if choice == "1":
                self.handle_assessment_workflow()
            elif choice == "2":
                print("\nExiting system. Goodbye.")
                break
            else:
                self.view.display_error("Invalid menu selection. Please choose 1 or 2.")

    def handle_assessment_workflow(self) -> None:
        """Executes the step-by-step risk assessment workflow."""
        # 1. Show available patients
        valid_ids = self.model.get_all_ids()
        self.view.display_patient_ids(valid_ids)
        
        # 2. Get and validate target Patient ID
        id_input = self.view.prompt_patient_id()
        if not id_input.isdigit():
            self.view.display_error("Patient ID must be a numeric integer value.")
            return
            
        patient_id = int(id_input)
        patient_metrics = self.model.get_patient(patient_id)
        if not patient_metrics:
            self.view.display_error(f"Patient ID {patient_id} does not exist in the database.")
            return

        # 3. Present data profile
        self.view.display_profile(patient_id, patient_metrics)
        
        # 4. Handle conditional updates
        if self.view.prompt_modification_choice():
            updated_metrics = {}
            for metric, current_val in patient_metrics.items():
                updated_metrics[metric] = self.view.prompt_metric_update(metric, current_val)
            self.model.update_patient(patient_id, updated_metrics)
            patient_metrics = updated_metrics # Sync local copy

        # 5. Evaluate and display report
        score, category = self.service.evaluate_patient_risk(patient_metrics)
        self.view.display_diagnostic_report(patient_id, score, category)


# =====================================================================
# 3-TIER AUTOMATED TESTING SUITE
# =====================================================================
def run_automated_testing_suite():
    print("\n" + "="*60)
    print("         LAUNCHING SYSTEM AUTOMATED TESTING SUITE")
    print("="*60)
    
    # -----------------------------------------------------------------
    # TIER 1: UNIT TESTS
    # -----------------------------------------------------------------
    print("\n[TIER 1] Running Unit Tests...")
    
    # Test Data Cleaning (Median Imputation)
    model = PatientModel()
    # Explicitly verify patient 102 who had BMI = 0.0 initially
    p102 = model.get_patient(102)
    assert p102["BMI"] > 0.0, f"Unit Test Failed: Data cleaning failed to resolve 0 BMI. Found {p102['BMI']}"
    # Valid patient BMIs: 22.5, 29.1, 36.4 -> median is 29.1
    assert p102["BMI"] == 29.1, f"Unit Test Failed: Expected median BMI 29.1, got {p102['BMI']}"
    print(" ✓ Unit Test Pass: Data Cleaning & Imputation successful.")
    
    # Test Scoring Engine Scoring Range Logic
    service = ClinicalRiskService()
    assert service.calculate_metric_score("Glucose", 90.0) == 0    # Low Range
    assert service.calculate_metric_score("Glucose", 110.0) == 1   # Med Range
    assert service.calculate_metric_score("Glucose", 140.0) == 2   # High Range
    print(" ✓ Unit Test Pass: Metric rule range mapping rules perform correctly.")
    
    # Test Global Category Mapping
    assert service.evaluate_patient_risk({"Glucose": 90, "BMI": 22, "Age": 25, "BloodPressure": 110})[1] == "Low Risk"
    assert service.evaluate_patient_risk({"Glucose": 140, "BMI": 38, "Age": 60, "BloodPressure": 145})[1] == "High Risk"
    print(" ✓ Unit Test Pass: Clinical category aggregator evaluating predictably.")

    # -----------------------------------------------------------------
    # TIER 2: END-TO-END (E2E) TESTS
    # -----------------------------------------------------------------
    print("\n[TIER 2] Running End-to-End (E2E) Integration Journey...")
    
    # Mocking user interaction behavior via a programmatically controlled view pipeline
    class MockedE2EView(ConsoleView):
        def __init__(self):
            # Simulated inputs sequential steps:
            # Select menu choice 1 -> Select patient 101 -> Say No to modification
            self.inputs = ["1", "101", "n", "2"]
            self.outputs = []
            
        def display_main_menu(self) -> str:
            return self.inputs.pop(0)
        def display_patient_ids(self, ids): pass
        def prompt_patient_id(self) -> str:
            return self.inputs.pop(0)
        def display_profile(self, pid, met): pass
        def prompt_modification_choice(self) -> bool:
            return self.inputs.pop(0) == 'y'
        def display_diagnostic_report(self, patient_id, score, category):
            self.outputs.append((patient_id, score, category))

    mock_view = MockedE2EView()
    e2e_controller = RiskAssessmentController(PatientModel(), ClinicalRiskService(), mock_view)
    
    # Execute full workflow sequence programmatically
    e2e_controller.run()
    
    # Assert expected outputs caught inside presentation layer simulation hook
    assert len(mock_view.outputs) == 1, "E2E Test Failed: Evaluation sequence did not record pipeline output."
    pid, score, risk = mock_view.outputs[0]
    assert pid == 101 and risk == "Low Risk", f"E2E Test Failed: Unexpected data calculations inside pipeline sequence. ({pid}, {risk})"
    print(" ✓ E2E Test Pass: End-to-end multi-layer pipeline interaction completely functional.")

    # -----------------------------------------------------------------
    # TIER 3: PERFORMANCE TESTS
    # -----------------------------------------------------------------
    print("\n[TIER 3] Running Performance Speed Benchmarks...")
    
    perf_model = PatientModel()
    perf_service = ClinicalRiskService()
    test_metrics = perf_model.get_patient(104)
    
    iterations = 10_000
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        perf_service.evaluate_patient_risk(test_metrics)
        
    end_time = time.perf_counter()
    total_duration = end_time - start_time
    avg_duration_ms = (total_duration / iterations) * 1000
    
    print(f" ✓ Performance Test Pass: Processed {iterations:,} clinical evaluations in {total_duration:.4f} seconds.")
    print(f"   Mean Evaluation Latency: {avg_duration_ms:.6f} ms per patient dataset lookup.")
    print("\n" + "="*60)
    print("         ALL AUTOMATED TESTING TIERS PASSED SUCCESSFULLY")
    print("="*60 + "\n")


# =====================================================================
# SYSTEM ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    if RUN_TEST_SUITE:
        run_automated_testing_suite()
    else:
        if RUN_IN_CONSOLE:
            db_model = PatientModel()
            rules_service = ClinicalRiskService()
            ui_view = ConsoleView()
            app = RiskAssessmentController(model=db_model, service=rules_service, view=ui_view)
            app.run()
        else:
            # Initialize layers via dependency injection for Streamlit Web View
            db_model = PatientModel()
            rules_service = ClinicalRiskService()
            
            # Execute modular Streamlit renderer
            StreamlitView.render_ui(model=db_model, service=rules_service)
