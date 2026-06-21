# Sepsis Prediction: Data Preprocessing & Feature Engineering Pipeline

This document provides a detailed technical and theoretical explanation of the steps taken to transform raw ICU data into a format suitable for high-performance machine learning models (XGBoost, LightGBM, Random Forest).

---

## 1. Data Loading & Initial Cleaning
### What we handle:
The raw dataset consists of hourly ICU records across thousands of patients. Key challenges include non-standardized feature names, inconsistent recording intervals, and mixed data types.

### How we handle it:
*   **Alignment:** Data is sorted by `Patient_ID` and `ICULOS` (ICU Length of Stay) to ensure temporal consistency.
*   **Filtering:** Rows with missing target labels (`SepsisLabel`) are removed (if any).

### Impact:
*   **Before:** Out-of-order records that would break time-series rolling calculations.
*   **After:** A clean, chronologically sorted stream of clinical data for every patient.

---

## 2. Handling Missing Values (Clinical Imputation)
### What we handle:
Medical data is "sparse" by nature. Labs (like Lactate or WBC) are only measured every few hours or days, leading to `NaN` values in the hourly snapshots.

### How we handle it:
*   **Forward-Fill (LOCF - Last Observation Carried Forward):** If a lab was taken at Hour 2, we assume its value remains relevant for Hour 3 and 4 until a new measurement is taken.
*   **Median/Zero Imputation:** For vitals never recorded for a patient, we use the global population median.

### Theoretical Proof:
Forward-fill mimics clinical reality: a doctor makes decisions based on the *latest available* lab result until a new one arrives. Simple mean imputation would "leak" future knowledge or create medically impossible "average" patients.

---

## 3. The "HospAdmTime" Outlier Problem
### What we handle:
`HospAdmTime` represents the time (in hours) between hospital admission and ICU admission. Our data showed values ranging from **-5,366** (over 7 months) to **+24** hours. 
*   **The Issue:** The extreme skewness (Standard Deviation of 167 vs. Median of -6) causes linear scalers to "squash" 99% of patients into a tiny, indistinguishable range.

### How we handle it: **Symmetric Log Transformation**
We apply the formula: $y = \text{sign}(x) \cdot \log(1 + |x|)$
*   This handles the negative values (pre-ICU stay) while compressing the multi-thousand-hour outliers.

### Impact Comparison:
| Method | "Normal" Patient (-2h) | Outlier Patient (-5000h) | Ratio/Resolution |
| :--- | :--- | :--- | :--- |
| Raw Data | -2.0 | -5000.0 | 2500x difference (Model "blind" to small changes) |
| Min-Max | 0.999 | 0.000 | Difference is too small to split on |
| **Symmetric Log** | **-1.09** | **-8.51** | **Balanced scale allowing clear decision splits** |

---

## 4. Advanced Temporal Feature Engineering
Sepsis is not a static state; it is a **dynamic process of deterioration**. A single snapshot of Heart Rate (HR) is less informative than the *trend* of HR over the last 12 hours.

### Features Created:
1.  **Rolling Statistics (12h Mean, Min, Max):** Captures the "baseline" and "volatility" of vitals (HR, BP, Temp).
2.  **Trend Slopes (6h):** Calculates the rate of change. (e.g., Is Blood Pressure dropping rapidly?)
3.  **Hourly Deltas:** Capture sudden "shocks" to the system between consecutive hours.

### Why this is superior:
Standard tabular models have no memory. By adding these features, we give the model a **temporal context** window, allowing it to "see" the patient's trajectory. 

---

## 5. Handling Varying Time Lengths
### What we handle:
Patient A may have 5 hours of data; Patient B may have 200 hours.

### How we handle it:
*   **Grouped Splitting:** We use `GroupShuffleSplit`. This ensures that all rows for a single patient stay together in either Train or Validation. This prevents **Data Leakage** (the model "memorizing" a patient it has already seen).
*   **Patient-wise Normalization:** During evaluation, we calculate SHAP values at the patient level to ensure a patient with 200 hours doesn't have 40x more "vote" than a patient with 5 hours.

---

## 6. Exhaustive Feature Dictionary & Clinical Significance

The features used in this model are divided into four primary categories: Raw Clinical Data, Engineered Measurement Indicators, Clinical Scoring Systems, and Temporal Trend Features.

### A. Raw Clinical Data (Vitals & Labs)
These are the direct measurements from the ICU bedside monitors and laboratory results.

*   **Vitals (HR, SBP, MAP, Resp, O2Sat, Temp):**
    *   *Heart Rate (HR):* Sepsis often causes tachycardia (high HR) as the heart works harder to pump oxygenated blood to organs during systemic inflammation.
    *   *Mean Arterial Pressure (MAP) & Systolic BP (SBP):* These are indicators of circulatory stability. Low values (hypotension) suggest septic shock and poor organ perfusion.
    *   *Respiration Rate (Resp):* Tachypnea (fast breathing) is one of the earliest signs of sepsis as the body attempts to compensate for metabolic acidosis.
*   **Laboratories (Lactate, Creatinine, Bilirubin, WBC, Platelets, etc.):**
    *   *Lactate:* The most critical marker for septic shock. High levels indicate tissue hypoxia (lack of oxygen at the cellular level).
    *   *Creatinine:* A marker of kidney function. Sudden rises indicate Acute Kidney Injury (AKI), a common complication of sepsis.
    *   *White Blood Cell Count (WBC):* Indicators of immune response. Both very high (leukocytosis) and very low (leukopenia) counts are diagnostic for sepsis.

---

### B. "Measurement Frequency" Features (e.g., `is_measured`, `time_since_last`)
**What they are:** These features don't measure the *value* of a vital sign, but rather the *act* of taking the measurement.
**Why they were created:** In an ICU, clinical sampling is **not random**. 
*   *Rationale:* If a doctor orders a Lactate test every 2 hours instead of every 12, it's because they perceive the patient is deteriorating. 
*   *Predictive Power:* The frequency of measurement is a "proxy" for the doctor's intuition. High-frequency monitoring is itself a strong predictor of clinical instability.
*   *Impact:* By capturing the measurement patterns, the model learns not just from the numbers, but from the **clinical behavior** of the care team.

---

### C. Clinical "Flag" Features (e.g., `hr_flag`, `map_flag`)
**What they are:** Binary indicators (0 or 1) that trigger when a vital sign crosses a critical medical threshold (e.g., SBP < 90 mmHg).
**Why they were created:** While XGBoost can find these thresholds itself, explicitly providing them as "Flags" simplifies the decision boundary.
*   *Rationale:* It mimics clinical protocols (e.g., "SIRS criteria"). 
*   *Impact:* It allows the model to immediately recognize a "crisis state" without having to calculate the significance of a specific numeric drop relative to the baseline in every single tree.

---

### D. Clinical Scoring Systems (SOFA & qSOFA)
These are internationally recognized scoring systems used by doctors to define and track sepsis severity.

*   **SOFA (Sequential Organ Failure Assessment):**
    *   *Components:* Combined score from Respiratory (PaO2/FiO2), Nervous (Glasgow Coma Scale), Cardiovascular (MAP), Liver (Bilirubin), Coagulation (Platelets), and Renal (Creatinine).
    *   *Meaning:* A SOFA score increase of 2+ points indicates a significantly higher risk of mortality and organ failure.
    *   *Why include it:* It is the "Gold Standard" for sepsis definition. By calculating this hourly, we give the model a high-level summary of the patient's multi-organ health.
*   **qSOFA (Quick SOFA):**
    *   *Components:* (1) Low blood pressure (SBP ≤ 100), (2) High respiratory rate (≥ 22), (3) Altered mental status.
    *   *Meaning:* A rapid bedside tool to identify patients at high risk of poor outcomes.
    *   *Why include it:* It uses only "easy to measure" vitals, making it highly sensitive to sudden drops in patient stability.

---

### E. Advanced Temporal Features (The "Trajectory" Features)
*   **12h Rolling Mean/Max/Min:** These stabilize noisy data. A single "spike" in HR might be noise, but a rising 12h mean is a trend.
*   **6h Slope:** Measures the "velocity" of deterioration. A patient with a MAP of 65 who is *rising* is safer than one with a MAP of 70 who is *falling rapidly*.
*   **Deltas:** Captures the "shock" value—how much a vital changed in just 60 minutes.

---

## 7. Mathematical Proof: Why this combination works
By combining **Raw Values** (The State), **Flags** (The Crisis), **Scoring Systems** (The Clinical Protocol), and **Slopes** (The Trajectory), we cover every dimension of a patient's ICU stay. 
1.  **Non-Linearity:** XGBoost excels at finding the complex interactions between these (e.g., *IF* Lactate is rising *AND* SOFA is > 4 *THEN* risk increases exponentially).
2.  **Robustness:** If one sensor fails (e.g., noisy BP), the model can still rely on the "Measurement Frequency" or "qSOFA" to maintain a prediction.

## 8. Final Analysis: Two-Stage Cascade & SHAP Findings

The Two-Stage Cascade model reached a performance of **0.8180 ROC AUC** and **29.6 hours of lead time**. Analysis of the SHAP values reveals a significant shift in how the model perceives patient risk.

### Key Finding 1: Individualization is King (`Lactate_zscore`)
*   **The Result:** `Lactate_zscore` became the **#1 most important feature** in the entire model.
*   **The Meaning:** The model learned that the *absolute* lactate value is less important than how much a patient's lactate has **jumped relative to their own normal baseline**. 
*   **Impact:** This confirms that the Method 2 (Baseline Normalization) successfully captured the "Shock" signal that standard models often miss.

### Key Finding 2: Sieve Efficiency
*   The Stage 1 "Sieve" successfully reduced the search space by ~20%, allowing the Stage 2 "Verifier" to focus on the top 1 million difficult cases. 
*   However, features like `ICU Admission Source Unspecified` still rank in the top 5, suggesting a persistent "Administrative Bias" where the hospital system source is a proxy for severity.

### Key Finding 3: Vital Stability vs. Labs
*   Features like `Temp_adv_12h_max` and `Resp_adv_12h_mean` rank significantly higher than raw vitals.
*   **Theoretical Proof:** Sepsis is a slow-burn process. The model correctly identifies that a **sustained trend** (captured by 12h windows) is a more reliable predictor of sepsis than a single "flash" measurement which might just be patient movement or sensor noise.

### The "Clinical Core" for the Slim Model
Based on the SHAP dataset, the following 10 features represent the absolute highest-impact physiological signals:
1.  **Lactate_zscore** (Tissue Hypoxia Trend)
2.  **BUN** (Kidney/Metabolic Stress)
3.  **FiO2** (Oxygen Requirement)
4.  **Temp_adv_12h_max** (Persistent Fever/Systemic Inflammation)
5.  **SIRS_score** (Standard Clinical Criterion)
6.  **Hgb** (Anemic/Oxygen carrying capacity)
7.  **Potassium** (Electrolyte Imbalance)
8.  **Creatinine** (Renal Failure marker)
9.  **O2Sat_zscore** (Deterioration in Oxygenation)
10. **Resp_adv_12h_mean** (Sustained Tachypnea)

---

## 9. Conclusion
By removing the "cheating" features (`ICULOS`, `HospAdmTime`), the model was forced to become a true **Physiological Deterioration Detector**. While the raw metrics (Precision/Recall) appear lower than biased models, the **Lead Time of 29 hours** and the focus on **Z-scores** makes this model far more clinically valuable for actual early-warning in an ICU setting.
