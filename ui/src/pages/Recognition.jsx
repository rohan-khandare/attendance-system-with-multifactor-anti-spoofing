// // import React, { useRef, useState, useEffect } from "react";
// // import axios from "axios";

// // const Recognition = () => {
// //   const videoRef = useRef(null);
// //   const canvasRef = useRef(null);
// //   const [status, setStatus] = useState("Initializing...");
// //   const [name, setName] = useState("");
// //   const [color, setColor] = useState("orange");

// //   // 1️⃣ Start camera
// //   useEffect(() => {
// //     const startCamera = async () => {
// //       try {
// //         const stream = await navigator.mediaDevices.getUserMedia({ video: true });
// //         if (videoRef.current) {
// //           videoRef.current.srcObject = stream;
// //         }
// //       } catch (err) {
// //         console.error("Camera access denied:", err);
// //         setStatus("Camera access denied.");
// //       }
// //     };
// //     startCamera();
// //   }, []);

// //   // 2️⃣ Send frames to backend every second
// //   useEffect(() => {
// //     const interval = setInterval(() => {
// //       captureAndSendFrame();
// //     }, 1000);
// //     return () => clearInterval(interval);
// //   }, []);

// //   const captureAndSendFrame = async () => {
// //     const video = videoRef.current;
// //     const canvas = canvasRef.current;
// //     if (!video || !canvas) return;

// //     const ctx = canvas.getContext("2d");
// //     ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

// //     const frame = canvas.toDataURL("image/jpeg");

// //     try {
// //       const response = await axios.post("http://127.0.0.1:8000/recognize", { frame });
// //       const { status, name } = response.data;

// //       setStatus(status);
// //       setName(name || "Unknown");

// //       if (status === "real") setColor("green");
// //       else if (status === "spoof") setColor("red");
// //       else setColor("orange");
// //     } catch (err) {
// //       console.error("Error sending frame:", err);
// //     }
// //   };

// //   return (
// //     <div className="bg-white p-6 rounded-xl shadow-lg">
// //       <h2 className="text-2xl font-semibold mb-4">Live Recognition</h2>

// //       <div className="flex flex-col items-center gap-4">
// //         <div className="relative">
// //           <video
// //             ref={videoRef}
// //             autoPlay
// //             muted
// //             width="640"
// //             height="480"
// //             className="rounded-xl border border-gray-300"
// //           ></video>
// //           <canvas
// //             ref={canvasRef}
// //             width="640"
// //             height="480"
// //             className="hidden"
// //           ></canvas>

// //           {/* Status Overlay */}
// //           <div
// //             className={`absolute bottom-4 left-1/2 -translate-x-1/2 px-4 py-2 rounded-lg text-white font-semibold bg-${color}-600`}
// //           >
// //             {status.toUpperCase()}
// //           </div>
// //         </div>

// //         <div className="mt-4 text-lg text-gray-800">
// //           <strong>Recognized:</strong>{" "}
// //           <span className="text-blue-600">{name}</span>
// //         </div>
// //       </div>
// //     </div>
// //   );
// // };

// // export default Recognition;


// // import React, { useRef, useState, useEffect } from "react";
// // import axios from "axios";
// // import { ToastContainer, toast } from "react-toastify";
// // import "react-toastify/dist/ReactToastify.css";

// // const Recognition = () => {
// //   const videoRef = useRef(null);
// //   const canvasRef = useRef(null);
// //   const [status, setStatus] = useState("Initializing...");
// //   const [name, setName] = useState("");
// //   const [color, setColor] = useState("orange");
// //   const [challenge, setChallenge] = useState("");
// //   const [remaining, setRemaining] = useState(null);

// //   useEffect(() => {
// //     const startCamera = async () => {
// //       try {
// //         const stream = await navigator.mediaDevices.getUserMedia({ video: true });
// //         if (videoRef.current) videoRef.current.srcObject = stream;
// //       } catch (err) {
// //         console.error("Camera access denied:", err);
// //         setStatus("Camera access denied.");
// //       }
// //     };
// //     startCamera();
// //   }, []);

// //   useEffect(() => {
// //     const interval = setInterval(() => captureAndSendFrame(), 1000);
// //     return () => clearInterval(interval);
// //   }, []);

// //   const captureAndSendFrame = async () => {
// //     const video = videoRef.current;
// //     const canvas = canvasRef.current;
// //     if (!video || !canvas) return;
// //     const ctx = canvas.getContext("2d");
// //     ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
// //     const frame = canvas.toDataURL("image/jpeg");

// //     try {
// //       const response = await axios.post("http://127.0.0.1:8000/recognize", { frame });
// //       const { status, name, challenge, remaining } = response.data;

// //       setStatus(status);
// //       setName(name);
// //       setChallenge(challenge || "");
// //       setRemaining(remaining ? Math.ceil(remaining) : null);

// //       if (status === "real") {
// //         setColor("green");
// //         toast.success(`Attendance logged for ${name}!`);
// //       } else if (status === "challenge") {
// //         setColor("orange");
// //       } else if (status === "unknown") {
// //         setColor("red");
// //       }

// //     } catch (err) {
// //       console.error("Error sending frame:", err);
// //     }
// //   };

// //   return (
// //     <div className="bg-white p-6 rounded-xl shadow-lg relative">
// //       <h2 className="text-2xl font-semibold mb-4">Live Recognition</h2>

// //       <div className="flex flex-col items-center gap-4">
// //         <div className="relative">
// //           <video
// //             ref={videoRef}
// //             autoPlay
// //             muted
// //             width="640"
// //             height="480"
// //             className="rounded-xl border border-gray-300"
// //           ></video>
// //           <canvas ref={canvasRef} width="640" height="480" className="hidden"></canvas>

// //           {/* Challenge prompt */}
// //           {status === "challenge" && (
// //             <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-orange-600 text-white px-4 py-2 rounded-lg font-semibold">
// //               Please {challenge?.replace("_", " ")} ({remaining ?? 0}s)
// //             </div>
// //           )}

// //           {/* Status bar */}
// //           <div
// //             className={`absolute bottom-4 left-1/2 -translate-x-1/2 px-4 py-2 rounded-lg text-white font-semibold bg-${color}-600`}
// //           >
// //             {status.toUpperCase()}
// //           </div>
// //         </div>

// //         <div className="mt-4 text-lg text-gray-800">
// //           <strong>Recognized:</strong> <span className="text-blue-600">{name}</span>
// //         </div>
// //       </div>

// //       <ToastContainer position="top-right" autoClose={2000} hideProgressBar />
// //     </div>
// //   );
// // };

// // export default Recognition;











// import React, { useRef, useState, useEffect } from "react";
// import axios from "axios";
// import { ToastContainer, toast } from "react-toastify";
// import "react-toastify/dist/ReactToastify.css";

// const Recognition = () => {
//   const videoRef = useRef(null);
//   const canvasRef = useRef(null);
//   const [status, setStatus] = useState("Initializing...");
//   const [name, setName] = useState("");
//   const [color, setColor] = useState("orange");
//   const [challenge, setChallenge] = useState("");
//   const [remaining, setRemaining] = useState(null);
//   const [loggedUsers, setLoggedUsers] = useState(new Set());

//   useEffect(() => {
//     const startCamera = async () => {
//       try {
//         const stream = await navigator.mediaDevices.getUserMedia({ video: true });
//         if (videoRef.current) videoRef.current.srcObject = stream;
//       } catch (err) {
//         console.error("Camera access denied:", err);
//         setStatus("Camera access denied.");
//       }
//     };
//     startCamera();
//   }, []);

//   useEffect(() => {
//     const interval = setInterval(() => captureAndSendFrame(), 1000);
//     return () => clearInterval(interval);
//   }, []);

//   const captureAndSendFrame = async () => {
//     const video = videoRef.current;
//     const canvas = canvasRef.current;
//     if (!video || !canvas) return;
//     const ctx = canvas.getContext("2d");
//     ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
//     const frame = canvas.toDataURL("image/jpeg");

//     try {
//       const response = await axios.post("http://127.0.0.1:8000/recognize", { frame });
//       const { status, name, challenge, remaining } = response.data;

//       setStatus(status);
//       setName(name || "Unknown");
//       setChallenge(challenge || "");
//       setRemaining(remaining ? Math.ceil(remaining) : null);

//       // if (status === "challenge") setColor("orange");
//       if (status === "challenge") {
//         setColor("orange");
//         if (challenge) {
//           setStatus(`Please ${challenge.replace("_", " ")} (${remaining ?? 0}s)`);
//         }
//       }

//       else if (status === "real") setColor("green");
//       else if (status === "spoof") setColor("red");
//       else setColor("gray");

//       // prevent duplicate toast
//       if (status === "real" && name && !loggedUsers.has(name)) {
//         toast.success(`Attendance logged for ${name}!`);
//         setLoggedUsers(prev => new Set(prev).add(name));
//       }

//     } catch (err) {
//       console.error("Error sending frame:", err);
//     }
//   };

//   return (
//     <div className="bg-white p-6 rounded-xl shadow-lg relative">
//       <h2 className="text-2xl font-semibold mb-4">Live Recognition</h2>

//       <div className="flex flex-col items-center gap-4">
//         <div className="relative">
//           <video
//             ref={videoRef}
//             autoPlay
//             muted
//             width="640"
//             height="480"
//             className="rounded-xl border border-gray-300"
//           ></video>
//           <canvas ref={canvasRef} width="640" height="480" className="hidden"></canvas>

//           {/* Challenge prompt */}
//           {status === "challenge" && challenge && (
//               <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-orange-600 text-white px-4 py-2 rounded-lg font-semibold shadow-lg">
//                 Please {challenge.replace("_", " ")} ({remaining ?? 0}s)
//               </div>
//             )}


//           {/* Status bar */}
//           <div
//             className={`absolute bottom-4 left-1/2 -translate-x-1/2 px-4 py-2 rounded-lg text-white font-semibold bg-${color}-600`}
//           >
//             {status.toUpperCase()}
//           </div>
//         </div>

//         <div className="mt-4 text-lg text-gray-800">
//           <strong>Recognized:</strong>{" "}
//           <span className="text-blue-600">{name}</span>
//         </div>
//       </div>

//       <ToastContainer position="top-right" autoClose={2500} hideProgressBar />
//     </div>
//   );
// };

// export default Recognition;










// src/pages/Recognition.jsx
import React, { useRef, useState, useEffect } from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const CHALLENGE_EMOJI = {
  blink: "👁️ Blink",
  smile: "😀 Smile",
  turn_left: "↩️ Turn Left",
  turn_right: "↪️ Turn Right"
};

export default function Recognition() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [status, setStatus] = useState("initializing");
  const [name, setName] = useState("");
  const [color, setColor] = useState("gray");
  const [challenge, setChallenge] = useState(null);
  const [remaining, setRemaining] = useState(0);
  const [loggedUsers, setLoggedUsers] = useState(new Set());

  useEffect(() => {
    const startCamera = async () => {
      try {
        const s = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) videoRef.current.srcObject = s;
      } catch (e) {
        console.error("Camera denied", e);
        setStatus("camera_denied");
      }
    };
    startCamera();
  }, []);

  useEffect(() => {
    const id = setInterval(() => captureAndSendFrame(), 900); // ~1fps (tune as needed)
    return () => clearInterval(id);
    // eslint-disable-next-line
  }, []);

  const captureAndSendFrame = async () => {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !c) return;
    const ctx = c.getContext("2d");
    c.width = 640;
    c.height = 480;
    ctx.drawImage(v, 0, 0, c.width, c.height);
    const frame = c.toDataURL("image/jpeg");

    try {
      const res = await axios.post("http://127.0.0.1:8000/recognize", { frame });
      const { status: s, name: n, challenge: ch, remaining: rem } = res.data;

      setStatus(s || "unknown");
      setName(n || "");
      setChallenge(ch || null);
      setRemaining(rem || 0);

      if (s === "challenge") {
        setColor("orange");
      } else if (s === "real") {
        setColor("green");
      } else if (s === "spoof") {
        setColor("red");
      } else {
        setColor("gray");
      }

      // toast attendance once per user
      if (s === "real" && n && !loggedUsers.has(n)) {
        toast.success(`Attendance logged for ${n}`);
        setLoggedUsers(prev => new Set(prev).add(n));
      }
    } catch (err) {
      console.error("Error sending frame:", err);
    }
  };

  return (
    <div className="bg-white p-4 rounded-xl shadow relative">
      <h2 className="text-2xl font-semibold mb-3">Live Recognition</h2>

      <div className="flex flex-col items-center gap-3">
        <div className="relative">
          <video ref={videoRef} autoPlay muted className="rounded-lg border" width="640" height="480" />
          <canvas ref={canvasRef} className="hidden" />

          {/* Challenge banner */}
          {status === "challenge" && challenge && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-orange-600 text-white px-4 py-2 rounded-lg font-semibold shadow-lg">
              {CHALLENGE_EMOJI[challenge] ?? `Please ${challenge}`} ({remaining}s)
            </div>
          )}

          {/* Status bar */}
          <div className={`absolute bottom-4 left-1/2 -translate-x-1/2 px-4 py-2 rounded-lg text-white font-semibold ${color === "green" ? "bg-green-600" : color === "orange" ? "bg-orange-600" : color === "red" ? "bg-red-600" : "bg-gray-600"}`}>
            {String(status).toUpperCase()}
          </div>
        </div>

        <div className="mt-2 text-lg text-gray-800">
          <strong>Recognized:</strong> <span className="text-indigo-600">{name || "—"}</span>
        </div>
      </div>

      <ToastContainer position="top-right" autoClose={2200} hideProgressBar />
    </div>
  );
}
