import { Link, useLocation } from "react-router-dom";
import { Home, UserPlus, Camera, CalendarDays, User } from "lucide-react";

const links = [
  { path: "/", label: "Dashboard", icon: <Home size={20} /> },
  { path: "/enroll", label: "Enroll", icon: <UserPlus size={20} /> },
  { path: "/recognition", label: "Mark Attendance", icon: <Camera size={20} /> },
  // { path: "/attendance", label: "Attendance Log", icon: <CalendarDays size={20} /> },
//   { path: "/profile", label: "Profile", icon: <User size={20} /> },
];

const Sidebar = () => {
  const { pathname } = useLocation();

  return (
    <aside className="w-64 bg-white border-r p-4">
      <h2 className="text-2xl font-bold mb-6 text-center text-blue-600">Smart Attend</h2>
      <nav className="space-y-2">
        {links.map(({ path, label, icon }) => (
          <Link
            key={path}
            to={path}
            className={`flex items-center gap-3 px-3 py-2 rounded-lg transition ${
              pathname === path
                ? "bg-blue-100 text-blue-600 font-medium"
                : "text-gray-700 hover:bg-gray-100"
            }`}
          >
            {icon} {label}
          </Link>
        ))}
      </nav>
    </aside>
  );
};

export default Sidebar;
