const imageUpload = document.querySelector("#imageUpload");
var uploaded_image = "";
imageUpload.addEventListener("change",function(){
  const reader = new FileReader();
  reader.addEventListener("load",()=>{
    uploaded_image = reader.result;
    document.querySelector("#display_image").style.backgroundImage = `url(${uploaded_image})`;

  });
  reader.readAsDataURL(this.files[0]);
})