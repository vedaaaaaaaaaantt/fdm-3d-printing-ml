import { useCallback, useEffect, useState } from "react";
import {
  BrainCircuit,
  RefreshCw,
  AlertCircle,
  TrendingUp,
  Info,
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const DEFAULT_MATERIAL = "PLA+";

function formatNumber(value) {
  return Number(value).toLocaleString("en-IN", {
    maximumFractionDigits: 2,
  });
}

function ExplainabilityCard() {
  const [materials, setMaterials] = useState([DEFAULT_MATERIAL]);
  const [selectedMaterial, setSelectedMaterial] = useState(DEFAULT_MATERIAL);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [materialsError, setMaterialsError] = useState("");

  const loadMaterials = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/materials`);

      if (!response.ok) {
        throw new Error(`Materials API returned ${response.status}`);
      }

      const result = await response.json();

      if (!Array.isArray(result.materials) || result.materials.length === 0) {
        throw new Error("No materials were found in the dataset.");
      }

      setMaterials(result.materials);

      const defaultMatch = result.materials.find(
        (item) => item.toLowerCase() === DEFAULT_MATERIAL.toLowerCase()
      );

      setSelectedMaterial((current) =>
        result.materials.some((item) => item === current)
          ? current
          : defaultMatch || result.materials[0]
      );

      setMaterialsError("");
    } catch (err) {
      setMaterialsError(err.message || "Unable to load materials.");
    }
  }, []);

  const loadExplainability = useCallback(async (material) => {
    setLoading(true);
    setError("");

    try {
      const query = new URLSearchParams({ material });

      const response = await fetch(
        `${API_BASE_URL}/api/pla-explainability?${query.toString()}`
      );

      if (!response.ok) {
        let detail = `API returned ${response.status}`;

        try {
          const body = await response.json();
          if (body.detail) detail = body.detail;
        } catch {
          // Keep the HTTP status message if the response is not JSON.
        }

        throw new Error(detail);
      }

      const result = await response.json();

      if (!Array.isArray(result.features)) {
        throw new Error("Unexpected SHAP response format.");
      }

      const features = result.features
        .map((item) => ({
          parameter: item.parameter,
          value: Number(item.mean_absolute_shap),
        }))
        .filter(
          (item) =>
            item.parameter &&
            Number.isFinite(item.value) &&
            item.value >= 0
        )
        .sort((a, b) => b.value - a.value);

      if (features.length === 0) {
        throw new Error(`No valid SHAP values were returned for ${material}.`);
      }

      setData({ ...result, features });
    } catch (err) {
      setData(null);
      setError(err.message || "Unable to load SHAP analysis.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadMaterials();
  }, [loadMaterials]);

  useEffect(() => {
    if (selectedMaterial) {
      loadExplainability(selectedMaterial);
    }
  }, [selectedMaterial, loadExplainability]);

  const chartData = data?.features ?? [];
  const highest = chartData[0];

  return (
    <section className="explainability-section">
      <div className="explainability-header">
        <div className="explainability-heading-copy">
          <p className="section-label">EXPLAINABLE AI · SHAP ANALYSIS</p>

          <h2>Why does the model make these predictions?</h2>

          <p>
            Explore which printing parameters have the greatest influence on
            predicted Peak Stress for samples of the selected material.
          </p>
        </div>

        <div className="explainability-icon">
          <BrainCircuit size={23} />
        </div>
      </div>

      <div className="shap-material-control">
        <label htmlFor="shap-material-select">Analyze material</label>

        <select
          id="shap-material-select"
          value={selectedMaterial}
          onChange={(event) => setSelectedMaterial(event.target.value)}
          disabled={materials.length === 0}
        >
          {materials.map((material) => (
            <option key={material} value={material}>
              {material}
            </option>
          ))}
        </select>

        {materialsError && (
          <p className="shap-inline-error">{materialsError}</p>
        )}
      </div>

      {loading && (
        <div className="shap-loading">
          <RefreshCw size={20} className="loading-icon" />
          <span>
            Loading feature-importance analysis for {selectedMaterial}…
          </span>
        </div>
      )}

      {!loading && error && (
        <div className="shap-error">
          <AlertCircle size={19} />
          <span>{error}</span>
          <button onClick={() => loadExplainability(selectedMaterial)}>
            Retry
          </button>
        </div>
      )}

      {!loading && !error && data && highest && (
        <>
          <div className="shap-highlight">
            <div className="shap-highlight-icon">
              <TrendingUp size={22} />
            </div>

            <div className="shap-highlight-content">
              <span className="shap-overline">
                MOST INFLUENTIAL PARAMETER
              </span>

              <h3>{highest.parameter}</h3>

              <p>
                Mean absolute SHAP:{" "}
                <strong>{formatNumber(highest.value)} kPa</strong>
              </p>
            </div>

            <div className="shap-rank" aria-label="Rank 1">
              #1
            </div>
          </div>

          <div className="shap-chart-container">
            <div className="shap-chart-heading">
              <div>
                <h3>Parameter Importance</h3>
                <p>Mean absolute SHAP values</p>
              </div>

              <span className="shap-dataset-tag">
                {data.material} dataset
              </span>
            </div>

            <div className="shap-chart">
              <ResponsiveContainer width="100%" height={350}>
                <BarChart
                  data={chartData}
                  layout="vertical"
                  margin={{
                    top: 8,
                    right: 24,
                    left: 12,
                    bottom: 8,
                  }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    horizontal={false}
                    stroke="#e5e9f0"
                  />

                  <XAxis
                    type="number"
                    tick={{ fontSize: 12, fill: "#687386" }}
                    axisLine={false}
                    tickLine={false}
                    tickFormatter={(value) =>
                      Number(value).toLocaleString("en-IN")
                    }
                  />

                  <YAxis
                    type="category"
                    dataKey="parameter"
                    width={155}
                    tick={{
                      fontSize: 12,
                      fill: "#4d586b",
                    }}
                    axisLine={false}
                    tickLine={false}
                  />

                  <Tooltip
                    formatter={(value) => [
                      `${formatNumber(value)} kPa`,
                      "Mean absolute SHAP",
                    ]}
                    contentStyle={{
                      borderRadius: 10,
                      border: "1px solid #e5e9f0",
                      boxShadow: "0 4px 16px rgba(23, 32, 51, 0.08)",
                      fontSize: 12,
                    }}
                    cursor={{ fill: "rgba(91, 141, 239, 0.08)" }}
                  />

                  <Bar
                    dataKey="value"
                    name="Mean absolute SHAP"
                    fill="#5B8DEF"
                    radius={[0, 5, 5, 0]}
                    barSize={22}
                    isAnimationActive
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="shap-interpretation">
            <div className="shap-interpretation-title">
              <Info size={17} />
              <strong>How to interpret this chart</strong>
            </div>

            <p>
              Larger mean absolute SHAP values indicate that a parameter has
              greater overall influence on the model’s predictions across the
              analyzed {data.material} samples. Values are in the model’s Peak
              Stress output scale (kPa).
            </p>
          </div>

          <div className="shap-caution">
            <AlertCircle size={17} />
            <p>
              {data.caution ||
                "SHAP importance describes model contributions, not physical causality. It does not show whether changing a parameter will increase or decrease Peak Stress."}
            </p>
          </div>

          <p className="shap-footer">
            SHAP TreeExplainer
            {" · "}Optimized XGBoost
            {" · "}{data.material}
            {" · "}{data.samples_analyzed} samples analyzed
            {" · "}{data.transformed_features} transformed features
          </p>
        </>
      )}
    </section>
  );
}

export default ExplainabilityCard;