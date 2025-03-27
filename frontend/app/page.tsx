"use client";

import { useState } from "react";

export default function Home() {
  const [image, setImage] = useState<File | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setImage(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!image) return alert("Please select an image first!");

    const formData = new FormData();
    formData.append("file", image);

    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Upload failed");

      const data = await response.json();
      setResultImage(`data:image/jpeg;base64,${data.image}`);
    } catch (error) {
      console.error("Error uploading image:", error);
    }
    setLoading(false);
  };

  return (
    <main className="p-10">
      <h1 className="text-2xl font-bold mb-4">Pedestrian Detection</h1>

      <input type="file" accept="image/*" onChange={handleImageChange} />
      <button
        className="mt-4 px-4 py-2 bg-blue-500 text-white rounded"
        onClick={handleUpload}
        disabled={loading}
      >
        {loading ? "Processing..." : "Upload & Detect"}
      </button>

      {resultImage && (
        <div className="mt-4">
          <h2 className="text-lg font-semibold">Detection Result:</h2>
          <img src={resultImage} alt="Detected" className="mt-2 max-w-full" />
        </div>
      )}
    </main>
  );
}
