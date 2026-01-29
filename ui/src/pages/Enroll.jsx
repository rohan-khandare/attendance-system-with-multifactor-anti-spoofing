// // // src/pages/EnrollUser.jsx
// // import { useState, useRef } from "react";

// // export default function Enroll() {
// //   const [name, setName] = useState("");
// //   const [email, setEmail] = useState("");
// //   const [dept, setDept] = useState("");
// //   const [cameraOn, setCameraOn] = useState(false);
// //   const videoRef = useRef(null);

// //   const startCamera = async () => {
// //     setCameraOn(true);
// //     const stream = await navigator.mediaDevices.getUserMedia({ video: true });
// //     videoRef.current.srcObject = stream;
// //   };

// //   const handleEnroll = async () => {
// //   if (!name || !email || !dept) {
// //     alert("Please fill all fields before enrolling.");
// //     return;
// //   }

// //   const canvas = document.createElement("canvas");
// //   const ctx = canvas.getContext("2d");
// //   canvas.width = videoRef.current.videoWidth;
// //   canvas.height = videoRef.current.videoHeight;
// //   ctx.drawImage(videoRef.current, 0, 0);
// //   const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg"));

// //   const formData = new FormData();
// //   formData.append("name", name);
// //   formData.append("email", email);
// //   formData.append("department", dept);
// //   formData.append("image", blob, "face.jpg");

// //   const res = await fetch("http://127.0.0.1:8000/enroll", {
// //     method: "POST",
// //     body: formData,
// //   });

// //   const data = await res.json();
// //   if (res.ok) {
// //     alert(data.message);
// //   } else {
// //     alert(`Error: ${data.error}`);
// //   }
// // };


// //   return (
// //     <div className="min-h-screen flex items-center justify-center bg-gray-50">
// //       <div className="bg-white rounded-2xl shadow-lg p-8 w-full max-w-md">
// //         <h2 className="text-2xl font-semibold mb-6 text-gray-800">
// //           Enroll New User
// //         </h2>

// //         <div className="space-y-4">
// //           <input
// //             type="text"
// //             placeholder="Full Name"
// //             className="w-full border rounded-lg px-4 py-2"
// //             value={name}
// //             onChange={(e) => setName(e.target.value)}
// //           />
// //           <input
// //             type="email"
// //             placeholder="Email"
// //             className="w-full border rounded-lg px-4 py-2"
// //             value={email}
// //             onChange={(e) => setEmail(e.target.value)}
// //           />
// //           <input
// //             type="text"
// //             placeholder="Department"
// //             className="w-full border rounded-lg px-4 py-2"
// //             value={dept}
// //             onChange={(e) => setDept(e.target.value)}
// //           />

// //           <div className="border rounded-lg flex flex-col items-center justify-center p-4">
// //             {cameraOn ? (
// //               <video ref={videoRef} autoPlay className="w-full rounded-lg" />
// //             ) : (
// //               <button
// //                 onClick={startCamera}
// //                 className="bg-blue-600 text-white py-2 px-4 rounded-lg flex items-center gap-2"
// //               >
// //                 <span>📷</span> Start Camera
// //               </button>
// //             )}
// //           </div>

// //           <button
// //             onClick={handleEnroll}
// //             className="w-full bg-gray-400 text-white py-2 rounded-lg mt-4"
// //           >
// //             Enroll User
// //           </button>
// //         </div>
// //       </div>
// //     </div>
// //   );
// // }


// // src/pages/EnrollUser.jsx
// import { useState, useRef } from "react";
// import { ToastContainer, toast } from "react-toastify";
// import "react-toastify/dist/ReactToastify.css";

// export default function Enroll() {
//   const [name, setName] = useState("");
//   const [email, setEmail] = useState("");
//   const [dept, setDept] = useState("");
//   const [cameraOn, setCameraOn] = useState(false);
//   const videoRef = useRef(null);

//   const startCamera = async () => {
//     try {
//       setCameraOn(true);
//       const stream = await navigator.mediaDevices.getUserMedia({ video: true });
//       videoRef.current.srcObject = stream;
//     } catch (err) {
//       toast.error("Camera access denied!");
//     }
//   };

//   const handleEnroll = async () => {
//     if (!name || !email || !dept) {
//       toast.warning("Please fill all fields!");
//       return;
//     }

//     if (!videoRef.current) {
//       toast.error("Camera not initialized.");
//       return;
//     }

//     // Capture frame from camera
//     const canvas = document.createElement("canvas");
//     const ctx = canvas.getContext("2d");
//     canvas.width = videoRef.current.videoWidth || 640;
//     canvas.height = videoRef.current.videoHeight || 480;
//     ctx.drawImage(videoRef.current, 0, 0);

//     const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg"));

//     const formData = new FormData();
//     formData.append("name", name);
//     formData.append("email", email);
//     formData.append("department", dept);
//     formData.append("image", blob, "face.jpg");

//     try {
//       const res = await fetch("http://127.0.0.1:8000/enroll", {
//         method: "POST",
//         body: formData,
//       });

//       const data = await res.json();

//       if (res.ok) {
//         toast.success(data.message);
//         setName("");
//         setEmail("");
//         setDept("");
//       } else {
//         toast.error(data.error || "Enrollment failed!");
//       }
//     } catch (err) {
//       toast.error("Server error. Try again!");
//     }
//   };

//   return (
//     <div className="min-h-screen flex items-center justify-center bg-gray-100 px-4">
//       <div className="bg-white rounded-2xl shadow-lg p-6 w-full max-w-md">
//         <h2 className="text-2xl font-semibold mb-5 text-gray-800 text-center">
//           Enroll New User
//         </h2>

//         <div className="space-y-4">
//           <input
//             type="text"
//             placeholder="Full Name"
//             className="w-full border rounded-lg px-4 py-2 focus:outline-blue-500"
//             value={name}
//             onChange={(e) => setName(e.target.value)}
//           />

//           <input
//             type="email"
//             placeholder="Email"
//             className="w-full border rounded-lg px-4 py-2 focus:outline-blue-500"
//             value={email}
//             onChange={(e) => setEmail(e.target.value)}
//           />

//           <input
//             type="text"
//             placeholder="Department"
//             className="w-full border rounded-lg px-4 py-2 focus:outline-blue-500"
//             value={dept}
//             onChange={(e) => setDept(e.target.value)}
//           />

//           {/* Camera Box */}
//           <div className="border rounded-lg flex flex-col items-center justify-center p-4 bg-gray-50">
//             {cameraOn ? (
//               <video ref={videoRef} autoPlay className="w-full rounded-lg" />
//             ) : (
//               <button
//                 onClick={startCamera}
//                 className="bg-blue-600 hover:bg-blue-700 transition text-white py-2 px-4 rounded-lg"
//               >
//                 📷 Start Camera
//               </button>
//             )}
//           </div>

//           {/* Enroll Button */}
//           <button
//             onClick={handleEnroll}
//             className="w-full bg-green-600 hover:bg-green-700 text-white py-2 rounded-lg mt-2 transition"
//           >
//             Enroll User
//           </button>
//         </div>
//       </div>

//       {/* Global Toast Component */}
//       <ToastContainer position="top-right" autoClose={2500} hideProgressBar />
//     </div>
//   );
// }





















// src/pages/EnrollUser.jsx
import { useState, useRef } from "react";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

export default function Enroll() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [dept, setDept] = useState("");
  const [cameraOn, setCameraOn] = useState(false);
  const videoRef = useRef(null);

  const startCamera = async () => {
    setCameraOn(true);
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.current.srcObject = stream;
  };

  const handleEnroll = async () => {
    if (!name || !email || !dept) {
      toast.error("Please fill all fields before enrolling.");
      return;
    }
    // capture one frame from camera
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg"));

    const formData = new FormData();
    formData.append("name", name);
    formData.append("email", email);
    formData.append("department", dept);
    formData.append("image", blob, "face.jpg");

    try {
      const res = await fetch("http://127.0.0.1:8000/enroll", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        toast.success(data.message || "Enrolled successfully");
        // optional: clear form
        setName(""); setEmail(""); setDept("");
      } else {
        toast.error(data.error || "Enrollment failed");
      }
    } catch (err) {
      console.error(err);
      toast.error("Enrollment request failed");
    }
  };

  return (
    <div className="min-h-screen flex justify-center bg-gray-50 p-">
      <div className="bg-white rounded-2xl shadow-lg p-6 w-full max-w-md">
        <h2 className="text-2xl font-semibold mb-4 text-gray-800">Enroll New User</h2>
        <div className="space-y-3">
          <input value={name} onChange={(e)=>setName(e.target.value)} placeholder="Full Name" className="w-full border rounded-lg px-3 py-2"/>
          <input value={email} onChange={(e)=>setEmail(e.target.value)} placeholder="Email" className="w-full border rounded-lg px-3 py-2"/>
          <input value={dept} onChange={(e)=>setDept(e.target.value)} placeholder="Department" className="w-full border rounded-lg px-3 py-2"/>

          <div className="border rounded-lg flex flex-col items-center justify-center p-3">
            {cameraOn ? (
              <video ref={videoRef} autoPlay className="w-full rounded-lg" width="480" height="360" />
            ) : (
              <button onClick={startCamera} className="bg-indigo-600 text-white py-2 px-4 rounded-lg flex items-center gap-2">
                <span>📷</span> Start Camera
              </button>
            )}
          </div>

          <button onClick={handleEnroll} className="w-full bg-emerald-600 text-white py-2 rounded-lg mt-2 hover:bg-emerald-700 transition">
            Enroll User
          </button>
        </div>
      </div>
      <ToastContainer position="top-right" autoClose={2200} hideProgressBar />
    </div>
  );
}
