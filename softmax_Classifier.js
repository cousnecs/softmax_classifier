var x=[[0.00,0.00 ], [ 0.00124061 , 0.01002453], [ 0.00221855, 0.02007983], [ 0.01732319  ,0.02486324], [ 0.01706577, 0.03662303], [-0.00267777, 0.05043401], [ 0.00992257, 0.05978827], [ 0.01760741  ,0.0684797 ],
    [ 0.01553685,  0.07930039], [ 0.02089989 , 0.08847405], [ 0.05191508,  0.08664793], [ 0.04742307  ,0.10048249], [ 0.06900689,  0.09965153], [ 0.0499487 , 0.12144244],
    [ 0.07073434, 0.12245249],[ 0.08471813, 0.1256172 ],[ 0.11622894,  0.112297  ],[ 0.13271225,  0.10896901],[ 0.11256735,  0.14278111],[ 0.12445527  ,0.14609539],[ 0.14226922,  0.14342814]];
var y=new Array(x.length);
for(var g=0;g<x.length;g++){
    if( g % 2 == 0){
        y[g]= 0;
    }else{
        y[g]=1;
    }
}

var w=new Array();
for(var i=0,l=x[0].length;i<l;i++){
    w[i]=new Array();
    for(var j=0;j<3;j++){
        w[i][j]=randf(x[0].length,3) * 0.01;
    }
}

var b= zeros(w[0].length),
    step_size=1e-0,
    reg=1e-0,
    num_examples=x.length;
 


for(var q=0,v=100;q<v;q++){
    var loss=0;
    var scores=new Array();
    var x_bat;
    var y_bat;
    var dp=[];
    var yp=[];
    for(var i =0;i<10;i++){
        y_bat=y[Math.ceil(Math.random(x.length)*10)];
        x_bat=x[Math.ceil(Math.random(x.length)*10)];
        dp.push(x_bat);
        yp.push(y_bat);
    }
    
    
    for(var t=0;t<x.length;t++){
        scores[t]=zeros(w[0].length);
    }
    //perform dot product

    for(var k=0;k<dp.length;k++){
        //var X=x[k];
        for(var n=0;n<dp[0].length;n++){
            var r=w[n];
            for(var j=0;j<r.length;j++){
                scores[k][j] +=dp[k][n]*r[j] + b[j];
            }

        }
    }

    //sove exponential of scores
    var exp_v=new Array();
    for(var i=0;i<scores.length;i++){
        exp_v[i]=zeros(scores[i].length);
    }
    scores.forEach(function(index,i){
        var length=scores[0].length;
        for(var j=0;j<length;j++){
            exp_v[i][j]=Math.exp(scores[i][j]);
        }
    });

    //sum of exponential
    var sum=zeros(exp_v.length);
    for(var i=0;i<exp_v.length;i++){
        var exp=exp_v[i];
        for(var j=0;j<exp.length;j++){
            sum[i] +=exp[j];
        }
    }
    //deviding exp by sum
    var probs=new Array();
    for(var i=0;i<scores.length;i++){
        probs[i]=zeros(scores[i].length);
    }

    for(var i=0;i<exp_v.length;i++){
        var exps=exp_v[i];
        for(var j=0;j<exps.length;j++){
            probs[i][j]= exps[j] / sum[i];
        }
    }

    //make a log of class and corect class

    /*var cor=new Array();
    for(var i=0;i<y.length;i++){
        cor[i]=zeros(y.length);
    }*/
    var cory=new Array();
    for(var i=0;i<y.length;i++){
        cory[i]=zeros(y.length);
    }
    for(var i=0;i<yp.length;i++){
        var z=yp[i];
        for(var j=0,k=0;k<x.length,j<z.length;j++,k++){
            cory[i][j]+=-Math.log(probs[k][z[j]]);

        }
    }


    //find data loss cory / num_examples

    var cor_sum=0;
    for(var i=0;i < cory.length;i++){
        var c= cory[i];
        for(var j =0; j< c.length;j++){
            cor_sum += c[j]/ num_examples;
        }
    }
    //var data_loss = cor_sum / num_examples ;
    var data_loss= cor_sum ;
    //lopp thru w to find the sum
    var e=0;
    w.forEach(function(index){
        index.forEach(function(j){
            e += j * j ;
        });
    });
    var reg_loss =0.5 * reg * e;
    loss =data_loss + reg_loss;
   
    //print itratiyon
    if( q % 10 == 0){
      //  console.log( "iteration : %s loss: %s",q,loss);
    }
   var dscores=probs;
    for(var k=0;k<num_examples;k++){
        for(var j=0;j<yp.length;j++){
            var z=yp[j];
            for(var ti=0;ti<z.length;ti++) {
                var f = z[ti]
                dscores[k][f] -= 1;
            }
        }
    }
    dscores.forEach(function(index,i){
        var length=dscores[i].length;
        for(var j=0;j<length;j++){
            dscores[i][j] = dscores[i][j] / num_examples;

        }
    });
    var dw = dot(T(dp),dscores);
    var db=a_sum(dscores);
    for(var i=0;i<dw.length;i++){
        var r=dw[i];
        for(var j=0;j<r.length;j++){
            dw[i][j] += reg * w[i][j];
          // w[i][j] += - step_size * dw[i][j];
        }
    }
    for(var i=0;i<w.length;i++){
        var r=w[i];
        for(var j=0;j<r.length;j++){
            //dw[i][j] += reg * w[i][j];
            w[i][j] += - step_size * dw[i][j];
        }
    }
    
    for(var i=0;i<db.length;i++){
        b[i] += -step_size * db[i];
    }

}

sor = dotsum(x,w,b);
var max;
var arg=argmax(sor);
var arlen = arg.length;
var ar_sum=0;
for(var pr=0;pr< arlen;pr++){
    ar_sum += arg[pr];
}
console.log(arg,(ar_sum / arlen));




function argmax(sor) {
    var sum=0,
        num=[];
    for (var i = 0; i < sor.length; i++) {
        var sey = sor[i];
        for (var si = 0; si < sey.length - 2; si++) {
            max = si;

            for (var sj = si + 1; sj < sey.length - 1; sj++) {
                if (sey[sj] < sey[max]) {
                    max = sj;

                }
            }

        }

        if(max ==y[i]) sum +=max,num.push(max);


    }
    return num;



}


//}





function a_sum(a){
    var sum= zeros(a[0].length);
    for(var j=0;j<a[0].length;j++){
        for(var i=1;i< a.length +1;i++){
            sum[j] += a[i -1][j]
        }
    }
    return sum;
}

function dotsum(x,w,b) {
    var s = new Array();
    for (var i = 0; i < x.length; i++) {
        s[i] = zeros(w[0].length);
    }
    if (x[0].length == w.length) {

        for (var k = 0; k < x.length; k++) {
            var X = x[k];
            for (var i = 0; i < w.length; i++) {
                var r = w[i];
                for (var j = 0; j < r.length; j++) {
                    s[k][j] += X[i] * r[j] + b[j];
                }

            }
        }
        return s;

    }
    else {
        throw new Error("dimesion not thesame");
    }
}

function dot (x,w){
    var s=new Array();
    for(var i =0;i < x.length;i++){
        s[i]=zeros(w[0].length);
    }
    if(x[0].length == w.length){

        for(var k=0;k<x.length;k++){
            var X=x[k];
            for(var i=0;i<w.length;i++){
                var r=w[i];
                for(var j=0;j<r.length;j++){
                    s[k][j] +=X[i]*r[j]; //+ b[j];
                }

            }
        }
        return s;

    }
    else{
        throw "dimesion not thesame";
    }
}

function T(a){
    var s=new Array(a[0].length);
    for(var i =0;i<s.length;i++){
        s[i] =zeros(x.length);
    }
    for(var i =0;i<a[0].length;i++){
        var k = a[i];
        for(var j=0;j<a.length;j++){
            s[i][j]= a[j][i];
        }
    }
    return s;
}
function randf(a, b) {
    return Math.random()*(b-a)+a;
}
function zeros(n) {
    var arr= new Array(n);
    for(var i=0;i<n;i++) { arr[i]= 0; }
    return arr;
}
