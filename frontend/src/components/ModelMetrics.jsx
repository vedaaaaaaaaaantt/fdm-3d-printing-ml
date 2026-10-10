import { useState } from "react";
import {
  BarChart3,
  Target,
  Gauge,
  Activity,
  Weight,
} from "lucide-react";

const modelResults = {
  "Peak Load": {
    unit: "N",
    summary: {
      r2: 0.875267,
      mae: 40.448812,
      rmse: 55.426999,
    },
    cv: {
      model: "Optimized XGBoost",
      r2: 0.903339,
      std: 0.018767,
      mae: 41.797708,
      rmse: 57.001450,
    },
    models: [
      { name: "Linear Regression", r2: 0.831097, mae: 47.817757, rmse: 64.498479 },
      { name: "Random Forest", r2: 0.799945, mae: 50.087189, rmse: 70.194933 },
      { name: "XGBoost", r2: 0.875267, mae: 40.448812, rmse: 55.426999, best: true },
      { name: "SVR", r2: 0.712393, mae: 53.556320, rmse: 84.164816 },
      { name: "Optimized XGBoost", r2: 0.867850, mae: 41.589141, rmse: 57.051144 },
    ],
  },

  "Peak Stress": {
    unit: "kPa",
    summary: {
      r2: 0.872309,
      mae: 1858.186124,
      rmse: 2678.167160,
    },
    cv: {
      model: "Optimized XGBoost",
      r2: 0.908120,
      std: 0.021768,
      mae: 1871.683247,
      rmse: 2581.293151,
    },
    models: [
      { name: "Linear Regression", r2: 0.840220, mae: 2227.072772, rmse: 2995.842254 },
      { name: "Random Forest", r2: 0.822129, mae: 2211.639640, rmse: 3160.892390 },
      { name: "XGBoost", r2: 0.870407, mae: 1884.552266, rmse: 2698.032330 },
      { name: "SVR", r2: 0.152715, mae: 5469.392283, rmse: 6898.774189 },
      { name: "Optimized XGBoost", r2: 0.872309, mae: 1858.186124, rmse: 2678.167160, best: true },
    ],
  },

  "Elastic Modulus": {
    unit: "MPa",
    summary: {
      r2: 0.956414,
      mae: 132.769935,
      rmse: 209.086023,
    },
    cv: {
      model: "Optimized XGBoost",
      r2: 0.940249,
      std: 0.024856,
      mae: 162.336839,
      rmse: 234.812052,
    },
    models: [
      { name: "Linear Regression", r2: 0.907072, mae: 246.035815, rmse: 305.299036 },
      { name: "Random Forest", r2: 0.947367, mae: 147.515930, rmse: 229.763105 },
      { name: "XGBoost", r2: 0.956414, mae: 132.769935, rmse: 209.086023, best: true },
      { name: "SVR", r2: 0.467927, mae: 424.895889, rmse: 730.530931 },
      { name: "Optimized XGBoost", r2: 0.944974, mae: 146.223396, rmse: 234.928127 },
    ],
  },
};

function formatNumber(value) {
  return value.toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function MetricBox({ icon, label, value, unit }) {
  return (
    <div className="metric-box">
      <div className="metric-icon">{icon}</div>
      <div>
        <p>{label}</p>
        <strong>
          {value}
          {unit && <span>{unit}</span>}
        </strong>
      </div>
    </div>
  );
}

function ModelMetrics() {
  const [selectedProperty, setSelectedProperty] = useState("Peak Stress");
  const result = modelResults[selectedProperty];
  const { summary, models, cv, unit } = result;

  return (
    <section className="metrics-section">
      <div className="metrics-heading">
        <div>
          <p className="section-label">MODEL PERFORMANCE</p>
          <h2>Prediction Performance</h2>
          <p className="metrics-description">
            Compare five regression models on the held-out test set
            for each predicted mechanical property.
          </p>
        </div>
        <BarChart3 size={22} />
      </div>

      <div className="metrics-property-selector">
        <label htmlFor="metrics-property">Evaluate property</label>
        <select
          id="metrics-property"
          value={selectedProperty}
          onChange={(event) => setSelectedProperty(event.target.value)}
        >
          <option value="Peak Load">Peak Load (N)</option>
          <option value="Peak Stress">Peak Stress (kPa)</option>
          <option value="Elastic Modulus">Elastic Modulus (MPa)</option>
        </select>
      </div>

      <div className="metric-summary">
        <MetricBox
          icon={<Target size={19} />}
          label="Test R² Score"
          value={summary.r2.toFixed(4)}
        />
        <MetricBox
          icon={<Gauge size={19} />}
          label="Test MAE"
          value={formatNumber(summary.mae)}
          unit={` ${unit}`}
        />
        <MetricBox
          icon={
            selectedProperty === "Peak Load" ? (
              <Weight size={19} />
            ) : (
              <Activity size={19} />
            )
          }
          label="Test RMSE"
          value={formatNumber(summary.rmse)}
          unit={` ${unit}`}
        />
      </div>

      <div className="model-table-wrapper">
        <table className="model-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>R²</th>
              <th>MAE ({unit})</th>
              <th>RMSE ({unit})</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr
                key={model.name}
                className={model.best ? "best-model" : ""}
              >
                <td>
                  <div className="model-name">
                    {model.best && (
                      <span className="best-badge">BEST TEST</span>
                    )}
                    {model.name}
                  </div>
                </td>
                <td>{model.r2.toFixed(4)}</td>
                <td>{formatNumber(model.mae)}</td>
                <td>{formatNumber(model.rmse)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="metrics-cross-validation">
        <div>
          <p className="section-label">CROSS-VALIDATION</p>
          <h3>5-Fold Cross-Validation</h3>
          <p className="metrics-description">
            Training-data cross-validation results for the
            hyperparameter-tuned model.
          </p>
          <p className="metrics-description">
            Evaluated model: <strong>{cv.model}</strong>
          </p>
        </div>
        <div className="cross-validation-score">
          <strong>{cv.r2.toFixed(4)}</strong>
          <span>± {cv.std.toFixed(4)} R² standard deviation</span>
        </div>
      </div>

      <div className="metrics-description">
        <p>
          Cross-validation MAE: {formatNumber(cv.mae)} {unit}.
          {" "}Cross-validation RMSE: {formatNumber(cv.rmse)} {unit}.
        </p>
        <p>
          The test-set table compares all five models using the
          held-out test data. Cross-validation figures above refer
          to Optimized XGBoost and are reported separately; they
          should not be interpreted as the cross-validation scores
          of whichever model ranks first in the test-set table.
          Performance on unseen experimental data may differ.
        </p>
      </div>
    </section>
  );
}

export default ModelMetrics;
