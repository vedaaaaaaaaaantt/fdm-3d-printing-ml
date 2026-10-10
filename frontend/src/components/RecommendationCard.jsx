
import { useEffect, useState } from "react";
import {
  Sparkles,
  Printer,
  Layers3,
  Percent,
  Network,
  Gauge,
  FlaskConical,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const API_URL = `${API_BASE_URL}/api/optimize`;

function formatStress(value) {
  return Number(value).toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function RecommendationCard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function loadOptimization() {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          Accept: "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`Optimization API returned ${response.status}`);
      }

      const result = await response.json();

      if (result.status !== "success") {
        throw new Error("The optimization request was unsuccessful.");
      }

      setData(result);
    } catch (err) {
      setError(err.message || "Unable to load optimization results.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadOptimization();
  }, []);

  if (loading) {
    return (
      <section className="recommendation-section">
        <p className="section-label">PROCESS OPTIMIZATION</p>
        <h2>Optimizing PLA+ parameters…</h2>
        <p>Evaluating candidate printing configurations.</p>
      </section>
    );
  }

  if (error) {
    return (
      <section className="recommendation-section">
        <p className="section-label">PROCESS OPTIMIZATION</p>
        <h2>Unable to load recommendations</h2>
        <p>{error}</p>
        <button onClick={loadOptimization} disabled={loading}>
          <RefreshCw size={16} /> Retry optimization
        </button>
      </section>
    );
  }

  const ml = data.closed_printer_recommendation;
  const measured = data.best_experimental_configuration;

  const parameters = [
    {
      label: "Printer",
      value: ml.printer_type,
      icon: <Printer size={18} />,
    },
    {
      label: "Material",
      value: ml.material,
      icon: <FlaskConical size={18} />,
    },
    {
      label: "Layer Thickness",
      value: `${ml.layer_thickness} mm`,
      icon: <Layers3 size={18} />,
    },
    {
      label: "Infill Density",
      value: `${ml.infill_density}%`,
      icon: <Percent size={18} />,
    },
    {
      label: "Infill Pattern",
      value: ml.infill_pattern,
      icon: <Network size={18} />,
    },
    {
      label: "Printing Speed",
      value: `${ml.speed} mm/s`,
      icon: <Gauge size={18} />,
    },
  ];

  return (
    <section className="recommendation-section">
      <div className="recommendation-header">
        <div>
          <p className="section-label">PROCESS OPTIMIZATION</p>
          <h2>Recommended PLA+ Configuration</h2>
          <p className="recommendation-description">
            Live results from bounded optimization using the
            retrained XGBoost model.
          </p>
        </div>

        <div className="recommendation-icon">
          <Sparkles size={22} />
        </div>
      </div>

      <div className="recommendation-grid">
        {parameters.map((item) => (
          <div className="recommendation-item" key={item.label}>
            <div className="recommendation-item-icon">
              {item.icon}
            </div>
            <div>
              <p>{item.label}</p>
              <strong>{item.value}</strong>
            </div>
          </div>
        ))}
      </div>

      <div className="recommendation-result">
        <div>
          <p>Predicted Peak Stress</p>
          <strong>
            {formatStress(ml.predicted_peak_stress_kpa)} kPa
          </strong>
          <span>
            {formatStress(ml.predicted_peak_stress_mpa)} MPa
          </span>
        </div>

        <div className="recommendation-note">
          <span>ML recommendation · Closed printer</span>
          <p>
            {data.candidates_evaluated} configurations evaluated.
            Physical testing is required to validate this prediction.
          </p>
        </div>
      </div>

      <div className="recommendation-context">
        <strong>
          <CheckCircle2 size={17} /> Best measured experiment
        </strong>
        <span>
          {measured.printer_type} printer · {measured.layer_thickness} mm
          · {measured.infill_density}% infill · {measured.infill_pattern}
          {" "}· {measured.speed} mm/s
        </span>
        <strong>
          {formatStress(measured.measured_peak_stress_mpa)} MPa measured
        </strong>
        <p>
          This is the highest-performing PLA+ experiment in the
          supplied dataset, not a new physical test.
        </p>
      </div>

      <div className="recommendation-context">
        <strong>
          <AlertCircle size={17} /> Interpretation
        </strong>
        <span>
          The model's predicted optimum and the best measured
          experiment are reported separately. Predictions do not
          guarantee experimental performance.
        </span>
      </div>
    </section>
  );
}

export default RecommendationCard;
