import { Box, Activity, Cpu } from "lucide-react";

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <div className="brand-icon">
          <Box size={22} />
        </div>

        <div>
          <h2>FDM<span>ML</span></h2>
          <p>3D Printing Intelligence</p>
        </div>
      </div>

      <div className="navbar-status">
        <div className="status-dot"></div>
        <span>Model Online</span>
        <Cpu size={17} />
        <span>XGBoost</span>
      </div>
    </nav>
  );
}

export default Navbar;