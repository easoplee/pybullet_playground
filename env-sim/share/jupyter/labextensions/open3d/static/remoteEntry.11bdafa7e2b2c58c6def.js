var _JUPYTERLAB;(()=>{"use strict";var e,r,t,n,a,o,i,u,l,d,s,c,f,p,h,v,g,b,m={81:(e,r,t)=>{var n={"./index":()=>t.e(568).then((()=>()=>t(568))),"./extension":()=>Promise.all([t.e(360),t.e(261)]).then((()=>()=>t(261)))},a=(e,r)=>(t.R=r,r=t.o(n,e)?n[e]():Promise.resolve().then((()=>{throw new Error('Module "'+e+'" does not exist in container.')})),t.R=void 0,r),o=(e,r)=>{if(t.S){var n=t.S.default,a="default";if(n&&n!==e)throw new Error("Container initialization failed as it has already been initialized with a different share scope");return t.S[a]=e,t.I(a,r)}};t.d(r,{get:()=>a,init:()=>o})}},w={};function y(e){var r=w[e];if(void 0!==r)return r.exports;var t=w[e]={id:e,loaded:!1,exports:{}};return m[e].call(t.exports,t,t.exports,y),t.loaded=!0,t.exports}y.m=m,y.c=w,y.d=(e,r)=>{for(var t in r)y.o(r,t)&&!y.o(e,t)&&Object.defineProperty(e,t,{enumerable:!0,get:r[t]})},y.f={},y.e=e=>Promise.all(Object.keys(y.f).reduce(((r,t)=>(y.f[t](e,r),r)),[])),y.u=e=>e+"."+{261:"b4aec1cc535ce3a3b811",360:"ade2d22e497a8518c7e6",486:"26a4cba3bd965d368e68",568:"5798f64006bfd0f8c416",580:"9be297587bab4703648f"}[e]+".js?v="+{261:"b4aec1cc535ce3a3b811",360:"ade2d22e497a8518c7e6",486:"26a4cba3bd965d368e68",568:"5798f64006bfd0f8c416",580:"9be297587bab4703648f"}[e],y.g=function(){if("object"==typeof globalThis)return globalThis;try{return this||new Function("return this")()}catch(e){if("object"==typeof window)return window}}(),y.o=(e,r)=>Object.prototype.hasOwnProperty.call(e,r),e={},r="open3d:",y.l=(t,n,a,o)=>{if(e[t])e[t].push(n);else{var i,u;if(void 0!==a)for(var l=document.getElementsByTagName("script"),d=0;d<l.length;d++){var s=l[d];if(s.getAttribute("src")==t||s.getAttribute("data-webpack")==r+a){i=s;break}}i||(u=!0,(i=document.createElement("script")).charset="utf-8",i.timeout=120,y.nc&&i.setAttribute("nonce",y.nc),i.setAttribute("data-webpack",r+a),i.src=t),e[t]=[n];var c=(r,n)=>{i.onerror=i.onload=null,clearTimeout(f);var a=e[t];if(delete e[t],i.parentNode&&i.parentNode.removeChild(i),a&&a.forEach((e=>e(n))),r)return r(n)},f=setTimeout(c.bind(null,void 0,{type:"timeout",target:i}),12e4);i.onerror=c.bind(null,i.onerror),i.onload=c.bind(null,i.onload),u&&document.head.appendChild(i)}},y.nmd=e=>(e.paths=[],e.children||(e.children=[]),e),(()=>{y.S={};var e={},r={};y.I=(t,n)=>{n||(n=[]);var a=r[t];if(a||(a=r[t]={}),!(n.indexOf(a)>=0)){if(n.push(a),e[t])return e[t];y.o(y.S,t)||(y.S[t]={});var o=y.S[t],i="open3d",u=(e,r,t,n)=>{var a=o[e]=o[e]||{},u=a[r];(!u||!u.loaded&&(!n!=!u.eager?n:i>u.from))&&(a[r]={get:t,from:i,eager:!!n})},l=[];switch(t){case"default":u("lodash","4.17.21",(()=>y.e(486).then((()=>()=>y(486))))),u("open3d","0.13.0",(()=>y.e(568).then((()=>()=>y(568))))),u("webrtc-adapter","4.2.2",(()=>y.e(580).then((()=>()=>y(580)))))}return e[t]=l.length?Promise.all(l).then((()=>e[t]=1)):1}}})(),(()=>{var e;y.g.importScripts&&(e=y.g.location+"");var r=y.g.document;if(!e&&r&&(r.currentScript&&(e=r.currentScript.src),!e)){var t=r.getElementsByTagName("script");t.length&&(e=t[t.length-1].src)}if(!e)throw new Error("Automatic publicPath is not supported in this browser");e=e.replace(/#.*$/,"").replace(/\?.*$/,"").replace(/\/[^\/]+$/,"/"),y.p=e})(),t=e=>{var r=e=>e.split(".").map((e=>+e==e?+e:e)),t=/^([^-+]+)?(?:-([^+]+))?(?:\+(.+))?$/.exec(e),n=t[1]?r(t[1]):[];return t[2]&&(n.length++,n.push.apply(n,r(t[2]))),t[3]&&(n.push([]),n.push.apply(n,r(t[3]))),n},n=(e,r)=>{e=t(e),r=t(r);for(var n=0;;){if(n>=e.length)return n<r.length&&"u"!=(typeof r[n])[0];var a=e[n],o=(typeof a)[0];if(n>=r.length)return"u"==o;var i=r[n],u=(typeof i)[0];if(o!=u)return"o"==o&&"n"==u||"s"==u||"u"==o;if("o"!=o&&"u"!=o&&a!=i)return a<i;n++}},a=e=>{var r=e[0],t="";if(1===e.length)return"*";if(r+.5){t+=0==r?">=":-1==r?"<":1==r?"^":2==r?"~":r>0?"=":"!=";for(var n=1,o=1;o<e.length;o++)n--,t+="u"==(typeof(u=e[o]))[0]?"-":(n>0?".":"")+(n=2,u);return t}var i=[];for(o=1;o<e.length;o++){var u=e[o];i.push(0===u?"not("+l()+")":1===u?"("+l()+" || "+l()+")":2===u?i.pop()+" "+i.pop():a(u))}return l();function l(){return i.pop().replace(/^\((.+)\)$/,"$1")}},o=(e,r)=>{if(0 in e){r=t(r);var n=e[0],a=n<0;a&&(n=-n-1);for(var i=0,u=1,l=!0;;u++,i++){var d,s,c=u<e.length?(typeof e[u])[0]:"";if(i>=r.length||"o"==(s=(typeof(d=r[i]))[0]))return!l||("u"==c?u>n&&!a:""==c!=a);if("u"==s){if(!l||"u"!=c)return!1}else if(l)if(c==s)if(u<=n){if(d!=e[u])return!1}else{if(a?d>e[u]:d<e[u])return!1;d!=e[u]&&(l=!1)}else if("s"!=c&&"n"!=c){if(a||u<=n)return!1;l=!1,u--}else{if(u<=n||s<c!=a)return!1;l=!1}else"s"!=c&&"n"!=c&&(l=!1,u--)}}var f=[],p=f.pop.bind(f);for(i=1;i<e.length;i++){var h=e[i];f.push(1==h?p()|p():2==h?p()&p():h?o(h,r):!p())}return!!p()},i=(e,r)=>{var t=y.S[e];if(!t||!y.o(t,r))throw new Error("Shared module "+r+" doesn't exist in shared scope "+e);return t},u=(e,r)=>{var t=e[r];return Object.keys(t).reduce(((e,r)=>!e||!t[e].loaded&&n(e,r)?r:e),0)},l=(e,r,t)=>"Unsatisfied version "+r+" of shared singleton module "+e+" (required "+a(t)+")",d=(e,r,t,n)=>{var a=u(e,t);return o(n,a)||"undefined"!=typeof console&&console.warn&&console.warn(l(t,a,n)),c(e[t][a])},s=(e,r,t)=>{var a=e[r];return(r=Object.keys(a).reduce(((e,r)=>!o(t,r)||e&&!n(e,r)?e:r),0))&&a[r]},c=e=>(e.loaded=1,e.get()),p=(f=e=>function(r,t,n,a){var o=y.I(r);return o&&o.then?o.then(e.bind(e,r,y.S[r],t,n,a)):e(r,y.S[r],t,n,a)})(((e,r,t,n)=>(i(e,t),d(r,0,t,n)))),h=f(((e,r,t,n,a)=>{var o=r&&y.o(r,t)&&s(r,t,n);return o?c(o):a()})),v={},g={171:()=>h("default","webrtc-adapter",[1,4,2,2],(()=>y.e(580).then((()=>()=>y(580))))),337:()=>p("default","@jupyter-widgets/base",[,[1,4],[1,3],[1,2],[1,1,1],1,1,1]),431:()=>h("default","lodash",[1,4,17,4],(()=>y.e(486).then((()=>()=>y(486)))))},b={360:[171,337,431],568:[171,337,431]},y.f.consumes=(e,r)=>{y.o(b,e)&&b[e].forEach((e=>{if(y.o(v,e))return r.push(v[e]);var t=r=>{v[e]=0,y.m[e]=t=>{delete y.c[e],t.exports=r()}},n=r=>{delete v[e],y.m[e]=t=>{throw delete y.c[e],r}};try{var a=g[e]();a.then?r.push(v[e]=a.then(t).catch(n)):t(a)}catch(e){n(e)}}))},(()=>{var e={440:0};y.f.j=(r,t)=>{var n=y.o(e,r)?e[r]:void 0;if(0!==n)if(n)t.push(n[2]);else if(360!=r){var a=new Promise(((t,a)=>n=e[r]=[t,a]));t.push(n[2]=a);var o=y.p+y.u(r),i=new Error;y.l(o,(t=>{if(y.o(e,r)&&(0!==(n=e[r])&&(e[r]=void 0),n)){var a=t&&("load"===t.type?"missing":t.type),o=t&&t.target&&t.target.src;i.message="Loading chunk "+r+" failed.\n("+a+": "+o+")",i.name="ChunkLoadError",i.type=a,i.request=o,n[1](i)}}),"chunk-"+r,r)}else e[r]=0};var r=(r,t)=>{var n,a,[o,i,u]=t,l=0;for(n in i)y.o(i,n)&&(y.m[n]=i[n]);for(u&&u(y),r&&r(t);l<o.length;l++)a=o[l],y.o(e,a)&&e[a]&&e[a][0](),e[o[l]]=0},t=self.webpackChunkopen3d=self.webpackChunkopen3d||[];t.forEach(r.bind(null,0)),t.push=r.bind(null,t.push.bind(t))})();var E=y(81);(_JUPYTERLAB=void 0===_JUPYTERLAB?{}:_JUPYTERLAB).open3d=E})();