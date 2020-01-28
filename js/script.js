// To display the current year at the footer
let date = new Date();
let fullYear = date.getFullYear();
let footerSpan = document.getElementById("date");
footerSpan.innerText = fullYear;