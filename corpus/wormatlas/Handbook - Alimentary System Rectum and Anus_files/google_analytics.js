// google_analytics_full.js

// Dynamically create and append the Google script tag
var script = document.createElement('script');
script.async = true;
script.src = 'https://www.googletagmanager.com/gtag/js?id=G-5BR0VM332D';
document.head.appendChild(script);

// Once the script is loaded, initialize the Google Analytics code
script.onload = function() {
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'G-5BR0VM332D');
};
