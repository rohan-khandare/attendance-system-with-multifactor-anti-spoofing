const Navbar = () => {
  return (
    <header className="bg-white shadow p-4 flex justify-between items-center">
      <h1 className="text-lg font-semibold text-gray-700">Face Recognition Based Attendance System with Anti-Spoofing</h1>
      <button className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition">
        Logout
      </button>
    </header>
  );
};

export default Navbar;
