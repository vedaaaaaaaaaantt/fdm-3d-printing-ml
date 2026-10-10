
import {
  Activity,
  BrainCircuit,
  LoaderCircle,
  Zap,
  Weight,
  TrendingUp,
  Gauge,
} from "lucide-react";

function formatNumber(value, digits = 2) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "—";
  }

  return number.toLocaleString("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function PropertyResult({
  icon,
  label,
  value,
  unit,
  description,
  accent,
}) {
  return (
    <div
      className={`prediction-property-item ${accent || ""}`}
    >
      <div className="prediction-property-heading">
        <div className="prediction-property-icon">
          {icon}
        </div>

        <span>{label}</span>
      </div>

      <div className="prediction-property-value">
        <strong>{value}</strong>
        <span>{unit}</span>
      </div>

      <p className="prediction-property-description">
        {description}
      </p>
    </div>
  );
}

function PredictionCard({
  prediction,
  loading,
  error,
}) {
  if (loading) {
    return (
      <section className="prediction-card">
        <div className="prediction-header">
          <div>
            <p className="section-label">
              LIVE ML PREDICTION
            </p>

            <h2>
              Predicting Mechanical Properties
            </h2>
          </div>

          <div className="prediction-icon">
            <LoaderCircle
              size={22}
              className="prediction-spinner"
            />
          </div>
        </div>

        <div className="prediction-loading">
          <LoaderCircle
            size={34}
            className="prediction-spinner"
          />

          <p>
            Running three machine learning models...
          </p>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="prediction-card">
        <div className="prediction-header">
          <div>
            <p className="section-label">
              LIVE ML PREDICTION
            </p>

            <h2>Prediction unavailable</h2>
          </div>

          <div className="prediction-icon">
            <Activity size={22} />
          </div>
        </div>

        <div className="prediction-error">
          {error}
        </div>
      </section>
    );
  }

  if (!prediction) {
    return (
      <section className="prediction-card">
        <div className="prediction-header">
          <div>
            <p className="section-label">
              LIVE ML PREDICTION
            </p>

            <h2>
              Predicted Mechanical Properties
            </h2>
          </div>

          <div className="prediction-icon">
            <BrainCircuit size={22} />
          </div>
        </div>

        <div className="prediction-empty">
          <div className="prediction-empty-icon">
            <Zap size={26} />
          </div>

          <strong>Ready to predict</strong>

          <p>
            Enter your printing parameters to predict
            Peak Load, Peak Stress, and Elastic Modulus.
          </p>
        </div>
      </section>
    );
  }

  const peakLoad = Number(
    prediction.predicted_peak_load_n
  );

  const peakStressKpa = Number(
    prediction.predicted_peak_stress_kpa
  );

  const peakStressMpa = Number(
    prediction.predicted_peak_stress_mpa
  );

  const elasticModulus = Number(
    prediction.predicted_elastic_modulus_mpa
  );

  const predictionsValid = [
    peakLoad,
    peakStressKpa,
    peakStressMpa,
    elasticModulus,
  ].every(Number.isFinite);

  if (!predictionsValid) {
    return (
      <section className="prediction-card">
        <div className="prediction-header">
          <div>
            <p className="section-label">
              LIVE ML PREDICTION
            </p>

            <h2>Incomplete prediction response</h2>
          </div>

          <div className="prediction-icon">
            <Activity size={22} />
          </div>
        </div>

        <div className="prediction-error">
          The API response is missing one or more
          valid mechanical-property predictions.
          Please run the prediction again.
        </div>
      </section>
    );
  }

  return (
    <section className="prediction-card">
      {/* HEADER */}

      <div className="prediction-header">
        <div>
          <p className="section-label">
            LIVE ML PREDICTION
          </p>

          <h2>
            Predicted Mechanical Properties
          </h2>

          <p className="metrics-description">
            Three predicted outcomes for your selected
            FDM printing parameters.
          </p>
        </div>

        <div className="prediction-icon">
          <BrainCircuit size={22} />
        </div>
      </div>

      {/* THREE MECHANICAL-PROPERTY PREDICTIONS */}

      <div className="prediction-property-grid">
        <PropertyResult
          icon={<Weight size={19} />}
          label="Peak Load"
          value={formatNumber(peakLoad)}
          unit="N"
          description="Predicted maximum tensile force"
          accent="property-load"
        />

        <PropertyResult
          icon={<Activity size={19} />}
          label="Peak Stress"
          value={formatNumber(peakStressMpa, 3)}
          unit="MPa"
          description={`${formatNumber(
            peakStressKpa,
            2
          )} kPa predicted`}
          accent="property-stress"
        />

        <PropertyResult
          icon={<Gauge size={19} />}
          label="Elastic Modulus"
          value={formatNumber(elasticModulus)}
          unit="MPa"
          description="Predicted material stiffness"
          accent="property-modulus"
        />
      </div>

      {/* PRIMARY PEAK-STRESS INDICATOR */}

      <div className="prediction-result">
        <div className="prediction-primary-label">
          <TrendingUp size={17} />
          <span>PEAK STRESS · PRIMARY OPTIMIZATION TARGET</span>
        </div>

        <div className="prediction-value">
          <strong>
            {formatNumber(peakStressKpa)}
          </strong>

          <span>kPa</span>
        </div>

        <div className="prediction-mpa">
          {formatNumber(peakStressMpa, 3)} MPa
        </div>

        <div className="prediction-model">
          <span className="prediction-live-dot" />

          <span>
            Peak Load: {prediction.peak_load_model || "XGBoost"}
            {" · "}
            Peak Stress: {prediction.peak_stress_model || "Optimized XGBoost"}
            {" · "}
            Elastic Modulus: {prediction.elastic_modulus_model || "XGBoost"}
          </span>
        </div>
      </div>

      {/* PEAK-STRESS SCALE */}

      <div className="prediction-scale">
        <div className="prediction-scale-labels">
          <span>Lower Stress</span>
          <span>Higher Stress</span>
        </div>

        <div className="prediction-scale-track">
          <div
            className="prediction-scale-fill"
            style={{
              width: `${Math.min(
                Math.max(
                  (peakStressKpa / 55000) * 100,
                  4
                ),
                100
              )}%`,
            }}
          />
        </div>
      </div>

      {/* INTERPRETATION */}

      <div className="prediction-note">
        <strong>What these predictions mean</strong>

        <p>
          Peak Load estimates the maximum tensile force
          in newtons. Peak Stress estimates the predicted
          maximum stress, and Elastic Modulus estimates
          stiffness in MPa. Each value comes from its
          respective trained regression model.
        </p>

        <p>
          These are machine learning predictions, not
          physical test measurements. Experimental
          testing is required to validate them.
        </p>
      </div>
    </section>
  );
}

export default PredictionCard;
