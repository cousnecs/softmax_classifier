
/**
---softmax classifier
 @author oni stephen (stevoni)
**/
//create main global object
var Softmax_classifier={};


(function(global){
    "use strict";

    //check condotions
    var assert = function(condition,message){
        if(!condition){ throw message || "Assertion failed"};
    }

    //utility class for all function needed
    var Utility={
        argmax(score,y){
            //find the index of the maximum number in an array
            //using sort algorithm
            //return the maximum  which equall the label
            var sum=0,
                num=[];

            for(var i=0;i<score.length;i++){
                var index = score[i];
                for(var si= 0; si <index.length -2; si++ ){
                    var max = si;

                    for(var sj=si +1;sj < index.length -1;sj++){
                        if(index[sj] < index[max]){
                            max = sj;
                        }
                    }
                }
                if(max ==y[i]) sum +=max,num.push(max);
            }
            return num;
        },
        //return the sum of arrays in a column
        a_sum(a){
            var sum=this.zeros(a[0].length);
            for(var j=0;j< a[0].length;j++){
                for(var i=1;i< a.length +1;i++){
                    sum[j] +=a[i -1][j];
                }
            }
            return sum;
        },
        dotsum(x,w,b){
            //calculate dot product of x,w if b is not given
            var dot_score= new Array();// store the dot product
            for(var i=0;i< x.length; i++){
                dot_score[i]=this.zeros(w[0].length);
            }
            if(typeof b =="undefined"){//check if b is undefined
                if(x[0].length ==w.length){
                    for(var k=0;k<x.length;k++){
                        var X_copy=x[k]; // make a copy of x index 
                        for(var i=0;i< w.length; i++){
                            var w_copy=w[i];
                            for(var j=0; j<w_copy.length;j++){
                                dot_score[k][j] +=X_copy[i] * w_copy[j];
                            }
                        }
                        return dot_score;
                    }
                }
                else{
                    //make a sanity check on the dimension
                    assert(x[0].length==w.length,"dimension not the same");
                }
            }
            else{
                //calculate dot product if b is given
                if(x[0].length ==w.length){
                    for(var k=0;k<x.length;k++){
                        var X_copy=x[k];
                        for(var i=0;i< w.length; i++){
                            var w_copy=w[i];
                            for(var j=0; j<w_copy.length;j++){
                                dot_score[k][j] +=X_copy[i] * w_copy[j] + b[j];
                            }
                        }
                        return dot_score;
                    }
                }
                else{
                    assert(x[0].length==w.length,"dimension not the same");
                }

            }
        },
        transpose(a){
            //make a transpose matrix
            //e.g matrix of shape (2,3) can be transpose to shape (3,2)
            var s =new Array(a[0].length);// store the result
            for(var i=0;i<a[0].length;i++){
                s[i]=this.zeros(a.length);
            }
            for(var i=0;i<a[0].length;i++){
                for(var j=0;j<a.length;j++){
                    s[i][j] += a[j][i];
                }
            }
            return s;

        },

        randf(a,b){
            //generate randon number
            return Math.random() * (b-a)+ a;
        },
        zeros(n){
            //generate zero Typedarrays of float64Array
            //generated from Andrej karpathy @Karpathy
            if(typeof(n)==='undefined' || isNaN(n)){return [];}
            if(typeof ArrayBuffer === 'undefined'){
                var ar=new Array(n);
                for(var i =0;i<n;i++){ar[i] =0;}
                    return ar;
            }else{
                return new Float64Array(n);
            }
        }
    };
var util=Object.create(Utility);//create a prototype class of Utility class

//main class
Softmax_classifier=function(opt){

        var opt=opt || {};//if no value create a class
        this.x=opt.x;
        if(!this.x){ assert(typeof opt.x !="undefined","x not given");} //enure x is given
        
        this.y=opt.y;
        if(!this.y){assert(typeof opt.y !='undefined',"y not given");}

        this.num_examples=this.x.length;

        this.weight=new Array();
        for(var i=0,l=this.x[0].length;i<l;i++){
            this.weight[i]=new Array();
            for(var j=0;j<3;j++){
                this.weight[i][j] = util.randf(this.x[0].length,3) * 0.01;
            }
        }

        this.b=util.zeros(this.weight[0].length); // biases
        this.step_size=opt.step_size || 1e-0;
        this.reg=opt.reg || 1e-0; //reqularization can be increase note it increase loss
        this.iter=opt.iter || 200; //number of iteration
        this.batch_size=opt.batch_size || 10; //number of class to be trained per iteration

};
Softmax_classifier.prototype={
    loss:function(){
        //calculate the loss
        for(var q=0,v=this.iter;q < v;q++){
            var loss=0;
            
            //store the batches of train arrays
            var x_bat,y_bat,
                xb=[],
                yb=[];
                
            for(var i=0;i<this.batch_size;i++){
                y_bat=y[Math.ceil(Math.random(this.num_examples)*this.batch_size)];
                x_bat=x[Math.ceil(Math.random(this.num_examples)*this.batch_size)];
                xb.push(x_bat);
                yb.push(y_bat);
            }
            
            var scores=new Array(); // store the product of the dotproduct
            for(var t=0;t<xb.length;t++){
                scores[t]=util.zeros(this.weight[0].length);
            }
            //perform dot product
            for(var k=0;k<xb.length;k++){
                for(var n=0;n<xb[0].length;n++){
                    var r=this.weight[n];
                    for(var j=0;j<r.length;j++){
                        scores[k][j] +=xb[k][n] * r[j] + this.b[j];
                    }
                }
            }
            //console.log(xb[0].length,scores);
            //sove exponential of scores
            var exp_v =new Array();
            for(var i=0;i<scores.length;i++){
                exp_v[i]=util.zeros(scores[0].length);
            }
            scores.forEach(function(index,i){
                var length=scores[i].length;
                for(var j=0;j<length;j++){
                    exp_v[i][j]=Math.exp(scores[i][j]);//find exponential across each index
                }
            });

            //find sum of exponential
            var sum=util.zeros(exp_v.length);
            for(var i=0;i<exp_v.length;i++){
                var exp=exp_v[i];
                for(var j=0;j<exp.length;j++){
                    sum[i] +=exp[j];
                }
            }
            
            var probs=new Array();//store the normalized class score
            for(var i=0;i<scores.length;i++){
                probs[i]=util.zeros(scores[i].length);
            }

            //deviding exponential by the sum
            for(var i=0;i<exp_v.length;i++){
                var exps=exp_v[i];
                for(var j=0;j<exps.length;j++){
                    probs[i][j]=exps[j] / sum[i];
                }
            }

            //make log of class and correct class

            var correct_class=new Array();
            for(var i=0;i<this.y.length;i++){
                correct_class[i]=util.zeros(this.y.length);
            }
            for(var i=0;i<yb.length;i++){
                var z=yb[i];
                for(var j=0,k=0;k<this,x.length,j<z.length;j++,k++){
                    correct_class[i][j] += -Math.log(probs[k][z[j]]);
                }
            }

            //find data loss correct_class / num_examples

            var cor_sum=0;
            for(var i=0;i<correct_class.length;i++){
                var c= correct_class[i];
                for(var j=0;j<c.length;j++){
                    cor_sum +=c[j] / this.num_examples;

                }
                
            }

            var data_loss =cor_sum;

            //loop thru weight to find the sum of square
            var e=0;
            this.weight.forEach(function(index){
                index.forEach(function(j){
                    e += j * j;
                });
            });

            var reg_loss = 0.5 *this.reg * e;
            //find the loss 
            loss =data_loss + reg_loss;
            //console.log(util.transpose(xb),xb.length);
            //print iteeration and loss
            if(q % 10 ==0){ console.log("iteration :%s loss: %s",q,loss);}

            //backprop thru the scores and parameters
            var dscores=probs;
            for(var k=0;k<this.num_examples;k++){
                for(var j=0;j<yb.length;j++){
                    var z = yb[j];
                    for(var ti=0;ti<z.length;ti++){
                        var f= z[ti];
                        dscores[k][f] -=1;
                    }
                }
            }
            var ni=this.num_examples;
            dscores.forEach(function(index,i){
                var length = dscores[i].length;
                for(var j=0;j<length;j++){
                    dscores[i][j] = dscores[i][j] / ni;
                }
            });
            //backprop thru weight
            var f=util.transpose(xb)
            var dw =util.dotsum(f,dscores);

            //backprop thru biase

            var db = util.a_sum(dscores);

            for(var i=0;i<dw.length;i++){
                var r =dw[i];
                for(var j=0;j<r.length;j++){
                    dw[i][j] += this.reg * this.weight[i][j];
                }
            }
            //copy above  loop code to prevent collision and unwanted result
            for(var i=0;i<this.weight.length;i++){
                var r=this.weight[i];
                for(var j=0;j<r.length;j++){
                    this.weight[i][j] += - this.step_size * dw[i][j];
                }
            }
            for(var i=0; i<db.length;i++){
                this.b[i] += -this.step_size * db[i];
            }

    }

    },
    predict(){
        //predict and give accuracy
        var pred=util.dotsum(this.x,this.weight,this.b);
        var arg= util.argmax(pred,this.y);
        var arlen = arg.length;
        var ar_sum=0;
        for(var pr=0;pr<arlen;pr++){
            ar_sum +=arg[pr];
        }
        console.log(arg,(ar_sum / arlen));
    }
};
global.Softmax_classifier=Softmax_classifier; //export softmax_classifier class

})(Softmax_classifier)

//export the libary to windows, or to module in nodejs
(function(lib){
    "use strict";
    if(typeof module ==="undefined" || typeof module.exports ==="undefined"){
        window.Softmax_classifier=lib; // in ordinary browser
    }else{
        module.exports = lib; // in nodejs
    }
})()