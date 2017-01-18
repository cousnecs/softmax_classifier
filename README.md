# softmax_classifier
These code perform softmax loss on the output score
*Note*
-- These code is a replical of python code from Cs231n convolutional Neural Network by Karpathy
-- And it as been tested based on the small data(score) produce in lecture 2 on svm and sofmax by karpathy
--for more clarification and to see the python code check the link below
- [Convolutional Neural Network ](http://cs231n.github.io/linear-classify/#softmax)

```javascript
//first load the class
var c=new Softmax_classifier(opt);

//to train 
c.loss();

// to predict and check accuracy
c.predict();

```
# e.g
```javascript
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
var opt={x:x,y:y};
var c=new Softmax_classifier(opt);
c.loss();
c.predict();
```