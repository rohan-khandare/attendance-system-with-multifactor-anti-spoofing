import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Enroll from "./pages/Enroll";
import Recognition from "./pages/Recognition";
// import AttendanceLog from "./pages/AttendanceLog";
// import Profile from "./pages/Profile";
import Navbar from "./components/Navbar";
import Sidebar from "./components/Sidebar";

function App() {
  return (
    <Router>
      <div className="flex min-h-screen bg-gray-100">
        <Sidebar />
        <div className="flex-1 flex flex-col">
          <Navbar />
          <main className="flex-1 p-6">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/enroll" element={<Enroll />} />
              <Route path="/recognition" element={<Recognition />} />
              {/* <Route path="/attendance" element={<AttendanceLog />} /> */}
              {/* <Route path="/profile" element={<Profile />} /> */}
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;
