import { useState } from "react";
import { SlidersHorizontal, LoaderCircle } from "lucide-react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function ParameterForm({ onPrediction, onLoading, onError }) {
  const [formData, setFormData] = useState({
    printer_type: "closed",
    material: "PLA+",
    layer_thickness: 0.3,
    infill_density: 80,
    infill_pattern: "quarter cubic",
    speed: 70,
    width: 6.173333,
    thickness: 3.411667,
  });

  const [loading, setLoading] = useState(false);

  const updateField = (field, value) => {
    setFormData((previous) => ({
      ...previous,
      [field]: value,
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    setLoading(true);
    onLoading(true);
    onError("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/predict`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            ...formData,
            layer_thickness: Number(formData.layer_thickness),
            infill_density: Number(formData.infill_density),
            speed: Number(formData.speed),
            width: Number(formData.width),
            thickness: Number(formData.thickness),
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json();

        throw new Error(
          errorData.detail
            ? JSON.stringify(errorData.detail)
            : "Prediction request failed."
        );
      }

      const result = await response.json();

      onPrediction(result);
    } catch (error) {
      console.error("Prediction error:", error);

      onError(
        error.message ||
          "Unable to connect to the prediction server."
      );
    } finally {
      setLoading(false);
      onLoading(false);
    }
  };

  return (
    <section className="parameter-card">
      <div className="section-header">
        <div>
          <p className="section-label">MODEL INPUTS</p>
          <h2>Printing Parameters</h2>
        </div>

        <SlidersHorizontal size={21} />
      </div>

      <form onSubmit={handleSubmit}>
        <div className="form-grid">

          {/* Printer */}

          <div className="form-group">
            <label>Printer Type</label>

            <select
              value={formData.printer_type}
              onChange={(event) =>
                updateField(
                  "printer_type",
                  event.target.value
                )
              }
            >
              <option value="closed">Closed</option>
              <option value="open">Open</option>
            </select>
          </div>


          {/* Material */}

          <div className="form-group">
            <label>Material</label>

            <select
              value={formData.material}
              onChange={(event) =>
                updateField(
                  "material",
                  event.target.value
                )
              }
            >
              <option value="PLA+">PLA+</option>
              <option value="PLA">PLA</option>
              <option value="PETG">PETG</option>
              <option value="PLA phosphorus">
                PLA phosphorus
              </option>
              <option value="TPU">TPU</option>
              <option value="ABS">ABS</option>
              <option value="PLA-CF">PLA-CF</option>
              <option value="PLA-transparent">
                PLA-transparent
              </option>
              <option value="PP">PP</option>
              <option value="rPET">rPET</option>
            </select>
          </div>


          {/* Layer Thickness */}

          <div className="form-group">
            <label>Layer Thickness</label>

            <div className="input-with-unit">
              <select
                value={formData.layer_thickness}
                onChange={(event) =>
                  updateField(
                    "layer_thickness",
                    Number(event.target.value)
                  )
                }
              >
                <option value="0.1">0.1</option>
                <option value="0.2">0.2</option>
                <option value="0.3">0.3</option>
              </select>

              <span>mm</span>
            </div>
          </div>


          {/* Infill Density */}

          <div className="form-group">
            <label>Infill Density</label>

            <div className="input-with-unit">
              <select
                value={formData.infill_density}
                onChange={(event) =>
                  updateField(
                    "infill_density",
                    Number(event.target.value)
                  )
                }
              >
                <option value="20">20</option>
                <option value="50">50</option>
                <option value="80">80</option>
              </select>

              <span>%</span>
            </div>
          </div>


          {/* Infill Pattern */}

          <div className="form-group">
            <label>Infill Pattern</label>

            <select
              value={formData.infill_pattern}
              onChange={(event) =>
                updateField(
                  "infill_pattern",
                  event.target.value
                )
              }
            >
              <option value="quarter cubic">
                Quarter Cubic
              </option>

              <option value="Grid">Grid</option>

              <option value="Triangles">
                Triangles
              </option>

              <option value="Zigzag">
                Zigzag
              </option>

              <option value="Gyroid">
                Gyroid
              </option>

              <option value="Cubic">
                Cubic
              </option>

              <option value="Lines">
                Lines
              </option>

              <option value="Concentric">
                Concentric
              </option>

              <option value="Cross">
                Cross
              </option>

              <option value="Lightning">
                Lightning
              </option>

              <option value="Octet">
                Octet
              </option>

              <option value="Tri Hexagon">
                Tri Hexagon
              </option>

              <option value="Cross3D">
                Cross3D
              </option>

              <option value="Cubic Subdivision">
                Cubic Subdivision
              </option>
            </select>
          </div>


          {/* Printing Speed */}

          <div className="form-group">
            <label>Printing Speed</label>

            <div className="input-with-unit">
              <select
                value={formData.speed}
                onChange={(event) =>
                  updateField(
                    "speed",
                    Number(event.target.value)
                  )
                }
              >
                <option value="30">30</option>
                <option value="50">50</option>
                <option value="70">70</option>
              </select>

              <span>mm/s</span>
            </div>
          </div>


          {/* Width */}

          <div className="form-group">
            <label>Specimen Width</label>

            <div className="input-with-unit">
              <input
                type="number"
                value={formData.width}
                min="5.93"
                max="6.63"
                step="0.001"
                onChange={(event) =>
                  updateField(
                    "width",
                    event.target.value
                  )
                }
                required
              />

              <span>mm</span>
            </div>
          </div>


          {/* Thickness */}

          <div className="form-group">
            <label>Specimen Thickness</label>

            <div className="input-with-unit">
              <input
                type="number"
                value={formData.thickness}
                min="3.073"
                max="3.813"
                step="0.001"
                onChange={(event) =>
                  updateField(
                    "thickness",
                    event.target.value
                  )
                }
                required
              />

              <span>mm</span>
            </div>
          </div>

        </div>


        <button
          type="submit"
          className="predict-button"
          disabled={loading}
        >
          {loading ? (
            <>
              <LoaderCircle
                size={18}
                className="loading-icon"
              />

              Predicting...
            </>
          ) : (
            "Predict Mechanical Properties"
          )}
        </button>
      </form>
    </section>
  );
}

export default ParameterForm;