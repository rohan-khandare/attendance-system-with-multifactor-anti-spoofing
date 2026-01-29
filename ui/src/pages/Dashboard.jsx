// // import { useEffect, useState } from "react";
// // import axios from "axios";
// // import {
// //   BarChart,
// //   Bar,
// //   XAxis,
// //   YAxis,
// //   Tooltip,
// //   ResponsiveContainer,
// //   CartesianGrid,
// // } from "recharts";

// // const Dashboard = () => {
// //   const [attendance, setAttendance] = useState([]);
// //   const [todayAttendance, setTodayAttendance] = useState([]);
// //   const [todayCount, setTodayCount] = useState(0);
// //   const [totalCount, setTotalCount] = useState(0);
// //   const [chartData, setChartData] = useState([]);

// //   // Fetch attendance data
// //   useEffect(() => {
// //     const fetchData = async () => {
// //       try {
// //         const allRes = await axios.get("http://127.0.0.1:8000/attendance");
// //         const todayRes = await axios.get("http://127.0.0.1:8000/attendance/today");

// //         setAttendance(allRes.data);
// //         setTodayAttendance(todayRes.data);
// //         setTodayCount(todayRes.data.length);
// //         setTotalCount(allRes.data.length);

// //         // Build chart data for last 7 days
// //         const daily = {};
// //         allRes.data.forEach((row) => {
// //           const day = row.timestamp.split(" ")[0];
// //           daily[day] = (daily[day] || 0) + 1;
// //         });

// //         const sorted = Object.entries(daily)
// //           .sort((a, b) => new Date(a[0]) - new Date(b[0]))
// //           .slice(-7)
// //           .map(([date, count]) => ({ date, count }));

// //         setChartData(sorted);
// //       } catch (err) {
// //         console.error("Error fetching attendance:", err);
// //       }
// //     };
// //     fetchData();
// //   }, []);

// //   // 📥 DOWNLOAD Today’s Attendance Sheet as CSV
// //   const downloadTodayCSV = () => {
// //     if (todayAttendance.length === 0) {
// //       alert("No attendance recorded today.");
// //       return;
// //     }

// //     const header = "Name,Timestamp,Confidence\n";
// //     const rows = todayAttendance
// //       .map(
// //         (r) =>
// //           `${r.name},${r.timestamp},${(r.confidence || 0).toFixed(3)}`
// //       )
// //       .join("\n");

// //     const csvContent = header + rows;
// //     const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });

// //     const link = document.createElement("a");
// //     link.href = URL.createObjectURL(blob);

// //     const today = new Date().toISOString().split("T")[0];
// //     link.download = `attendance_${today}.csv`;

// //     link.click();
// //   };

// //   return (
// //     <div className="p-6 bg-gray-50 min-h-screen">
// //       <h2 className="text-3xl font-semibold mb-6 text-gray-800">
// //         📊 Attendance Dashboard
// //       </h2>

// //       {/* Summary Cards */}
// //       <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
        
// //         <div className="bg-white rounded-2xl shadow p-5 text-center">
// //           <p className="text-gray-500">Total Attendance</p>
// //           <h3 className="text-3xl font-bold text-blue-600">{totalCount}</h3>
// //         </div>

// //         <div className="bg-white rounded-2xl shadow p-5 text-center">
// //           <p className="text-gray-500">Today's Attendance</p>
// //           <h3 className="text-3xl font-bold text-green-600">{todayCount}</h3>

// //           <button
// //             onClick={downloadTodayCSV}
// //             className="mt-3 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition"
// //           >
// //             ⬇ Download Today’s CSV
// //           </button>
// //         </div>

// //         <div className="bg-white rounded-2xl shadow p-5 text-center">
// //           <p className="text-gray-500">Recognized Users</p>
// //           <h3 className="text-3xl font-bold text-indigo-600">
// //             {new Set(attendance.map((a) => a.name)).size}
// //           </h3>
// //         </div>

// //         <div className="bg-white rounded-2xl shadow p-5 text-center">
// //           <p className="text-gray-500">Avg Confidence</p>
// //           <h3 className="text-3xl font-bold text-orange-600">
// //             {attendance.length
// //               ? (
// //                   attendance.reduce(
// //                     (acc, r) => acc + (r.confidence || 0),
// //                     0
// //                   ) / attendance.length
// //                 ).toFixed(2)
// //               : 0}
// //           </h3>
// //         </div>
// //       </div>

// //       {/* Attendance Trend Chart */}
// //       <div className="bg-white rounded-2xl shadow p-6 mb-10">
// //         <h3 className="text-xl font-semibold mb-4 text-gray-700">
// //           Attendance Trend (Last 7 Days)
// //         </h3>

// //         <ResponsiveContainer width="100%" height={300}>
// //           <BarChart data={chartData}>
// //             <CartesianGrid strokeDasharray="3 3" />
// //             <XAxis dataKey="date" />
// //             <YAxis allowDecimals={false} />
// //             <Tooltip />
// //             <Bar dataKey="count" fill="#4F46E5" barSize={50} />
// //           </BarChart>
// //         </ResponsiveContainer>
// //       </div>

// //       {/* Recent Attendance Table */}
// //       <div className="bg-white rounded-2xl shadow p-6">
// //         <h3 className="text-xl font-semibold mb-4 text-gray-700">
// //           Recent Attendance Logs
// //         </h3>

// //         <div className="overflow-x-auto">
// //           <table className="min-w-full border border-gray-200">
// //             <thead className="bg-gray-100 text-gray-700">
// //               <tr>
// //                 <th className="py-2 px-4 text-left">Name</th>
// //                 <th className="py-2 px-4 text-left">Timestamp</th>
// //                 <th className="py-2 px-4 text-left">Confidence</th>
// //               </tr>
// //             </thead>

// //             <tbody>
// //               {attendance.slice(0, 10).map((row, idx) => (
// //                 <tr key={idx} className="border-t">
// //                   <td className="py-2 px-4">{row.name}</td>
// //                   <td className="py-2 px-4 text-gray-600">{row.timestamp}</td>
// //                   <td className="py-2 px-4 text-gray-600">
// //                     {(row.confidence || 0).toFixed(3)}
// //                   </td>
// //                 </tr>
// //               ))}
// //             </tbody>

// //           </table>
// //         </div>
// //       </div>
// //     </div>
// //   );
// // };

// // export default Dashboard;





// import { useEffect, useState } from "react";
// import axios from "axios";
// import {
//   BarChart,
//   Bar,
//   XAxis,
//   YAxis,
//   Tooltip,
//   ResponsiveContainer,
//   CartesianGrid,
// } from "recharts";

// const Dashboard = () => {
//   const [attendance, setAttendance] = useState([]);
//   const [todayAttendance, setTodayAttendance] = useState([]);
//   const [enrolledUsers, setEnrolledUsers] = useState([]);
//   const [showEnrolled, setShowEnrolled] = useState(false);

//   const [todayCount, setTodayCount] = useState(0);
//   const [totalCount, setTotalCount] = useState(0);
//   const [chartData, setChartData] = useState([]);

//   // Fetch attendance + enrolled users
//   useEffect(() => {
//     const fetchData = async () => {
//       try {
//         const allRes = await axios.get("http://127.0.0.1:8000/attendance");
//         const todayRes = await axios.get("http://127.0.0.1:8000/attendance/today");
//         const usersRes = await axios.get("http://127.0.0.1:8000/enrolled-users");

//         setAttendance(allRes.data);
//         setTodayAttendance(todayRes.data);
//         setTodayCount(todayRes.data.length);
//         setTotalCount(allRes.data.length);
//         setEnrolledUsers(usersRes.data);

//         // Build chart data for last 7 days
//         const daily = {};
//         allRes.data.forEach((row) => {
//           const day = row.timestamp.split(" ")[0];
//           daily[day] = (daily[day] || 0) + 1;
//         });

//         const sorted = Object.entries(daily)
//           .sort((a, b) => new Date(a[0]) - new Date(b[0]))
//           .slice(-7)
//           .map(([date, count]) => ({ date, count }));

//         setChartData(sorted);

//       } catch (err) {
//         console.error("Error fetching attendance:", err);
//       }
//     };

//     fetchData();
//   }, []);

//   // DOWNLOAD Today CSV
//   const downloadTodayCSV = () => {
//     if (todayAttendance.length === 0) {
//       alert("No attendance recorded today.");
//       return;
//     }

//     const header = "Name,Timestamp,Confidence\n";
//     const rows = todayAttendance
//       .map(
//         (r) => `${r.name},${r.timestamp},${(r.confidence || 0).toFixed(3)}`
//       )
//       .join("\n");

//     const csvContent = header + rows;
//     const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });

//     const link = document.createElement("a");
//     link.href = URL.createObjectURL(blob);

//     const today = new Date().toISOString().split("T")[0];
//     link.download = `attendance_${today}.csv`;

//     link.click();
//   };

//   // Helper: Get attendance stats per enrolled user
//   const getUserStats = (username) => {
//     const logs = attendance.filter((a) => a.name === username);
//     return {
//       count: logs.length,
//       last: logs.length ? logs[0].timestamp : "—",
//     };
//   };

//   return (
//     <div className="p-6 bg-gray-50 min-h-screen">
//       <h2 className="text-3xl font-semibold mb-6 text-gray-800">
//         Attendance Dashboard
//       </h2>

//       {/* Summary Cards */}
//       <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">

//         <div className="bg-white rounded-2xl shadow p-5 text-center">
//           <p className="text-gray-500">Total Attendance</p>
//           <h3 className="text-3xl font-bold text-blue-600">{totalCount}</h3>
//         </div>

//         <div className="bg-white rounded-2xl shadow p-5 text-center">
//           <p className="text-gray-500">Today's Attendance</p>
//           <h3 className="text-3xl font-bold text-green-600">{todayCount}</h3>

//           <button
//             onClick={downloadTodayCSV}
//             className="mt-3 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition"
//           >
//             ⬇ Download CSV
//           </button>
//         </div>

//         {/* Replace Recognized with Enrolled Users */}
//         <div className="bg-white rounded-2xl shadow p-5 text-center">
//           <p className="text-gray-500">Enrolled Users</p>

//           <h3
//             onClick={() => setShowEnrolled(!showEnrolled)}
//             className="text-3xl font-bold text-purple-600 cursor-pointer hover:opacity-70 transition"
//           >
//             {enrolledUsers.length}
//           </h3>

//           <p className="text-sm text-purple-500">Click to view users</p>
//         </div>

//         <div className="bg-white rounded-2xl shadow p-5 text-center">
//           <p className="text-gray-500">Avg Confidence</p>
//           <h3 className="text-3xl font-bold text-orange-600">
//             {attendance.length
//               ? (
//                   attendance.reduce((acc, r) => acc + (r.confidence || 0), 0) /
//                   attendance.length
//                 ).toFixed(2)
//               : 0}
//           </h3>
//         </div>
//       </div>

//       {/* SHOW ENROLLED USERS LIST */}
//       {showEnrolled && (
//         <div className="bg-white rounded-2xl shadow p-6 mb-10">
//           <h3 className="text-xl font-semibold mb-4 text-gray-700">
//             Enrolled Users & Attendance Summary
//           </h3>

//           <div className="overflow-x-auto">
//             <table className="min-w-full border border-gray-200">
//               <thead className="bg-gray-100 text-gray-700">
//                 <tr>
//                   <th className="py-2 px-4 text-left">Name</th>
//                   <th className="py-2 px-4 text-left">Total Attendance</th>
//                   <th className="py-2 px-4 text-left">Last Seen</th>
//                 </tr>
//               </thead>
//               <tbody>
//                 {enrolledUsers.map((user, idx) => {
//                   const stats = getUserStats(user);
//                   return (
//                     <tr key={idx} className="border-t">
//                       <td className="py-2 px-4">{user}</td>
//                       <td className="py-2 px-4 text-gray-700">{stats.count}</td>
//                       <td className="py-2 px-4 text-gray-600">{stats.last}</td>
//                     </tr>
//                   );
//                 })}
//               </tbody>
//             </table>
//           </div>
//         </div>
//       )}

//       {/* Attendance Trend Chart */}
//       <div className="bg-white rounded-2xl shadow p-6 mb-10">
//         <h3 className="text-xl font-semibold mb-4 text-gray-700">
//           Attendance Trend (Last 7 Days)
//         </h3>

//         <ResponsiveContainer width="100%" height={300}>
//           <BarChart data={chartData}>
//             <CartesianGrid strokeDasharray="3 3" />
//             <XAxis dataKey="date" />
//             <YAxis allowDecimals={false} />
//             <Tooltip />
//             <Bar dataKey="count" fill="#4F46E5" barSize={50} />
//           </BarChart>
//         </ResponsiveContainer>
//       </div>

//       {/* Recent Attendance Table */}
//       <div className="bg-white rounded-2xl shadow p-6">
//         <h3 className="text-xl font-semibold mb-4 text-gray-700">
//           Recent Attendance Logs
//         </h3>

//         <div className="overflow-x-auto">
//           <table className="min-w-full border border-gray-200">
//             <thead className="bg-gray-100 text-gray-700">
//               <tr>
//                 <th className="py-2 px-4 text-left">Name</th>
//                 <th className="py-2 px-4 text-left">Timestamp</th>
//                 <th className="py-2 px-4 text-left">Confidence</th>
//               </tr>
//             </thead>

//             <tbody>
//               {attendance.slice(0, 10).map((row, idx) => (
//                 <tr key={idx} className="border-t">
//                   <td className="py-2 px-4">{row.name}</td>
//                   <td className="py-2 px-4 text-gray-600">{row.timestamp}</td>
//                   <td className="py-2 px-4 text-gray-600">
//                     {(row.confidence || 0).toFixed(3)}
//                   </td>
//                 </tr>
//               ))}
//             </tbody>

//           </table>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default Dashboard;












import { useEffect, useState } from "react";
import axios from "axios";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

const Dashboard = () => {
  const [attendance, setAttendance] = useState([]);
  const [todayAttendance, setTodayAttendance] = useState([]);
  const [enrolledUsers, setEnrolledUsers] = useState([]);
  const [showEnrolled, setShowEnrolled] = useState(false);

  const [todayCount, setTodayCount] = useState(0);
  const [totalCount, setTotalCount] = useState(0);
  const [chartData, setChartData] = useState([]);

  // -------------------------------
  // FIX: Fetch Attendance + Users
  // -------------------------------
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [allRes, todayRes, usersRes] = await Promise.all([
          axios.get("http://127.0.0.1:8000/attendance"),
          axios.get("http://127.0.0.1:8000/attendance/today"),
          axios.get("http://127.0.0.1:8000/enrolled-users"),
        ]);

        const allLogs = allRes.data;

        // FIX: Always sort attendance newest → oldest
        allLogs.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

        setAttendance(allLogs);
        setTodayAttendance(todayRes.data);
        setTodayCount(todayRes.data.length);
        setTotalCount(allLogs.length);

        // FIX: Remove duplicates + sort alphabetically
        const cleanUsers = [...new Set(usersRes.data)].sort();
        setEnrolledUsers(cleanUsers);

        // Chart logic (last 7 days)
        const daily = {};
        allLogs.forEach((row) => {
          const day = row.timestamp.split(" ")[0];
          daily[day] = (daily[day] || 0) + 1;
        });

        const sorted = Object.entries(daily)
          .sort((a, b) => new Date(a[0]) - new Date(b[0]))
          .slice(-7)
          .map(([date, count]) => ({ date, count }));

        setChartData(sorted);
      } catch (err) {
        console.error("Dashboard error:", err);
      }
    };

    fetchData();
  }, []);

  // -------------------------------
  // Download CSV
  // -------------------------------
  const downloadTodayCSV = () => {
    if (todayAttendance.length === 0) {
      alert("No attendance recorded today.");
      return;
    }

    const header = "Name,Timestamp,Confidence\n";
    const rows = todayAttendance
      .map(
        (r) => `${r.name},${r.timestamp},${(r.confidence || 0).toFixed(3)}`
      )
      .join("\n");

    const blob = new Blob([header + rows], { type: "text/csv" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);

    const today = new Date().toISOString().split("T")[0];
    link.download = `attendance_${today}.csv`;
    link.click();
  };

  // -------------------------------
  // FIX: Proper Last Seen & Count
  // -------------------------------
  const getUserStats = (username) => {
    const logs = attendance.filter((x) => x.name === username);

    if (logs.length === 0) {
      return { count: 0, last: "—" };
    }

    return {
      count: logs.length,
      last: logs[0].timestamp, // newest after sort
    };
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h2 className="text-3xl font-semibold mb-6 text-gray-800">
        Attendance Dashboard
      </h2>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
        <div className="bg-white rounded-2xl shadow p-5 text-center">
          <p className="text-gray-500">Total Attendance</p>
          <h3 className="text-3xl font-bold text-blue-600">{totalCount}</h3>
        </div>

        <div className="bg-white rounded-2xl shadow p-5 text-center">
          <p className="text-gray-500">Today's Attendance</p>
          <h3 className="text-3xl font-bold text-green-600">{todayCount}</h3>

          <button
            onClick={downloadTodayCSV}
            className="mt-3 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition"
          >
            ⬇ Download CSV
          </button>
        </div>

        <div className="bg-white rounded-2xl shadow p-5 text-center">
          <p className="text-gray-500">Enrolled Users</p>

          <h3
            onClick={() => setShowEnrolled(!showEnrolled)}
            className="text-3xl font-bold text-purple-600 cursor-pointer hover:opacity-70 transition"
          >
            {enrolledUsers.length}
          </h3>
          <p className="text-sm text-purple-500">Click to view users</p>
        </div>

        <div className="bg-white rounded-2xl shadow p-5 text-center">
          <p className="text-gray-500">Avg Confidence</p>
          <h3 className="text-3xl font-bold text-orange-600">
            {attendance.length
              ? (
                  attendance.reduce((acc, r) => acc + (r.confidence || 0), 0) /
                  attendance.length
                ).toFixed(2)
              : 0}
          </h3>
        </div>
      </div>

      {/* Enrolled Users Table */}
      {showEnrolled && (
        <div className="bg-white rounded-2xl shadow p-6 mb-10">
          <h3 className="text-xl font-semibold mb-4 text-gray-700">
            Enrolled Users & Attendance Summary
          </h3>

          <table className="min-w-full border border-gray-200">
            <thead className="bg-gray-100 text-gray-700">
              <tr>
                <th className="py-2 px-4 text-left">Name</th>
                <th className="py-2 px-4 text-left">Total Attendance</th>
                <th className="py-2 px-4 text-left">Last Seen</th>
              </tr>
            </thead>

            <tbody>
              {enrolledUsers.map((user, idx) => {
                const stats = getUserStats(user);
                return (
                  <tr key={idx} className="border-t">
                    <td className="py-2 px-4">{user}</td>
                    <td className="py-2 px-4">{stats.count}</td>
                    <td className="py-2 px-4">{stats.last}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Chart */}
      <div className="bg-white rounded-2xl shadow p-6 mb-10">
        <h3 className="text-xl font-semibold mb-4 text-gray-700">
          Attendance Trend (Last 7 Days)
        </h3>

        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" fill="#4F46E5" barSize={50} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Recent Attendance */}
      <div className="bg-white rounded-2xl shadow p-6">
        <h3 className="text-xl font-semibold mb-4 text-gray-700">
          Recent Attendance Logs
        </h3>

        <table className="min-w-full border">
          <thead className="bg-gray-100 text-gray-700">
            <tr>
              <th className="py-2 px-4 text-left">Name</th>
              <th className="py-2 px-4 text-left">Timestamp</th>
              <th className="py-2 px-4 text-left">Confidence</th>
            </tr>
          </thead>

          <tbody>
            {attendance.slice(0, 10).map((row, idx) => (
              <tr key={idx} className="border-t">
                <td className="py-2 px-4">{row.name}</td>
                <td className="py-2 px-4">{row.timestamp}</td>
                <td className="py-2 px-4">
                  {(row.confidence || 0).toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Dashboard;
