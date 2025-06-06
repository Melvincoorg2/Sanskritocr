document.addEventListener("DOMContentLoaded", function () {
  const fileInput = document.getElementById("imageUpload");

  // Preview the selected image
  fileInput.addEventListener("change", function (event) {
    const reader = new FileReader();
    reader.onload = function () {
      const preview = document.getElementById("imagePreview");
      preview.src = reader.result;
      preview.style.display = "block";

      // Hide annotated image on new selection
      document.getElementById("annotatedImage").style.display = "none";
    };
    reader.readAsDataURL(event.target.files[0]);
  });
});

function analyzeImage() {
  const fileInput = document.getElementById("imageUpload");
  const file = fileInput.files[0];

  if (!file) {
    alert("Please select an image first.");
    return;
  }

  const formData = new FormData();
  formData.append("image", file);

  // Set initial messages
  document.getElementById("result").innerText = "Processing...";
  document.getElementById("result_k").innerText = "Processing...";
  document.getElementById("annotatedImage").style.display = "none";

  fetch("http://localhost:8000/process-image", {
    method: "POST",
    body: formData
  })
    .then(async (res) => {
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Server returned ${res.status}: ${errorText}`);
      }
      return res.json();
    })
    .then((data) => {
      console.log("✅ Received data:", data);

      // Update result texts
      document.getElementById("result").innerText = data.result || "No result received.";
      document.getElementById("result_k").innerText = data.kannada_result || "No Kannada result received.";

      // Show annotated image if available
      if (data.annotated_image_base64) {
        const annotatedImage = document.getElementById("annotatedImage");
        annotatedImage.src = `data:image/jpeg;base64,${data.annotated_image_base64}`;
        annotatedImage.style.display = "block";
      }
    })
    .catch((err) => {
      console.error("⚠️ Error:", err);
      document.getElementById("result").innerText = "An error occurred: " + err.message;
      document.getElementById("result_k").innerText = "";
    });
}
