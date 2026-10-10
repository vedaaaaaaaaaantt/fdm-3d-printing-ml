
import { useState } from "react";

import Navbar from "./components/Navbar";
import ParameterForm from "./components/ParameterForm";
import PredictionCard from "./components/PredictionCard";
import ModelMetrics from "./components/ModelMetrics";
import RecommendationCard from "./components/RecommendationCard";
import ExplainabilityCard from "./components/ExplainabilityCard";

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handlePrediction = (result) => {
    setPrediction(result);
  };

  const handleLoading = (value) => {
    setLoading(value);
  };

  const handleError = (message) => {
    setError(message);

    if (message) {
      setPrediction(null);
    }
  };

  return (
    <>
      <Navbar />

      <main className="app-container">
        {/* HERO */}

        <section className="hero">
          <div>
            <p className="eyebrow">
              FDM PROCESS INTELLIGENCE
            </p>

            <h1>
              Predict & Optimize
              <br />
              <span>
                3D Printing Performance
              </span>
            </h1>

            <p className="hero-description">
              Machine learning-based prediction of Peak Load,
              Peak Stress, and Elastic Modulus for FDM-printed
              components using printing process and material
              parameters.
            </p>
          </div>
        </section>

        {/* INPUT PARAMETERS AND LIVE PREDICTIONS */}

        <div className="dashboard-grid">
          <ParameterForm
            onPrediction={handlePrediction}
            onLoading={handleLoading}
            onError={handleError}
          />

          <PredictionCard
            prediction={prediction}
            loading={loading}
            error={error}
          />
        </div>

        {/* MODEL PERFORMANCE */}

        <ModelMetrics />

        {/* EXPLAINABLE AI */}

        <ExplainabilityCard />

        {/* PROCESS OPTIMIZATION */}

        <RecommendationCard />
      </main>
    </>
  );
}

export default App;
