/*
* To change this license header, choose License Headers in Project Properties.
* To change this template file, choose Tools | Templates
* and open the template in the editor.
*/


import java.io.BufferedWriter;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Pattern;

import JFlex.Out;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * This class is the main class of the DSFS algorithm, which consists of three components,
 * including intra-feature value outlierness computing (i.e., the delta function in our ICDM2016 paper),
 * the adjacent matrix of the feature graph, and dense subgraph discovery of the feature graph.
 * Note that we skip the value graph construction and go directly to construct the feature graph, 
 * which can speed up the algorithm a bit.
 * @author Guansong Pang 
 */
public class SubgraphDiscovery2{
    
    private ArrayList<ArrayList<Double>> fMatrix = new ArrayList<ArrayList<Double>>();
    private ArrayList<Double> featIndWgts = new ArrayList<Double>();   //to store total weights for each feature
    private ArrayList<Integer> featStatus = new ArrayList<Integer>();   //to record whether the feature has been removed or not
    private ArrayList<Integer> featIndice = new ArrayList<Integer>();   //to record whether the feature has been removed or not
    
    /**
     * the main method for calling Charikar greedy, sequential backward and Las Vegas based
     * dense subgraph discovery
     * @param cpList the list of coupled centroids: each centroid contains the co-occurrence frequency of each value with other values
     * @return the IDs of features to be removed
     */
    public String denseSubgraphDiscovery(ArrayList<CoupledValueCentroid> cpList, Instances data, int dataSize) {
    	String[] attarray = new String[data.numAttributes()];
    	double[][] dataMatrix = new double[data.numInstances()][data.numAttributes()];
    	// 处理非数值型特征
    	for(int i = 0; i < data.numAttributes(); i++) {
//    		System.out.println(data.attribute(j));
    		String attr = data.attribute(i).toString();
    		String attrvalue = attr.substring(attr.indexOf("{")+1, attr.indexOf("}"));
//    		int attvnum = (attrvalue.length()+1)/2;
//    		System.out.print(" length:"+attvnum+" ");
    		attarray[i] = attrvalue;
//    		System.out.println(attarray[i]);
    	}
        for(int j = 0; j < data.numInstances(); j++) {
        	Instance inst = data.instance(j);
//            System.out.println(inst);
            for(int k = 0; k < inst.numAttributes(); k++) {
            	String str = inst.stringValue(k).toString();
              	if(!str.matches("^[0-9]*$")) {
              	str = String.valueOf(attarray[k].indexOf(str,1)/2);
              	}
//            System.out.print(str+",");
              	
            Double val = Double.parseDouble(str);
//            System.out.print(val+",");
            dataMatrix[j][k] = val;
            }
        }
        
    	double[] fo = calcIntraFeatureWeight(cpList,dataSize);
//        adjacentMatrix(cpList);
        featureAdjacentMatrix(cpList,fo);
        ArrayList<String> discardFeats = new ArrayList<String>();
        double[] den = charikarGreedySearchforFeatGraph(discardFeats, dataMatrix);
//        System.out.println(discardFeats);
   
//        double min = Double.MAX_VALUE;
//        int minID = -1;
//
//        double avg_den = 0.0;
//        for(int m = 0; m < den.length; m++) {
//        	avg_den += den[m];
//        }
//        avg_den = avg_den / den.length * 0.6;
////        System.out.println("avg_den:"+avg_den);
////        avg_den = avg_den / den.length * 1.2;
////        System.out.println();
//        label:for(int n = 0; n < den.length; n++) {
////        	System.out.print(den[n]+",");
//        	if(den[n] < avg_den) {
//        		min = den[n];
//        		minID = n;
//        		break label;
//        	}
//        }
        
        double max = -Double.MAX_VALUE;
        int maxID = -1;
        double avg_den = 0.0;
        for(int m = 0; m < den.length; m++) {
        	avg_den += den[m];
//        	System.out.println("den["+m+"]:"+den[m]);
        }
        avg_den = avg_den / den.length;
        System.out.println("avg_den:"+avg_den);
//        avg_den = avg_den / den.length * 0.99;
//        System.out.print("length:"+den.length);
        label:for(int n = 0; n < den.length; n++) {
//        	System.out.print(den[n]+",");
        	if(den[n] > avg_den) {
        		max = den[n];
        		maxID = n;
        		break label;
        	}
        }
        
//        System.out.println();
//    	}
        System.out.println("Mid:"+(new DecimalFormat("#0.0000")).format(max)+" ");
//        System.out.println("minID:"+minID);
        Plot.plotYPoints(den, 3, DSFS4ODUtils.dataSetName, DSFS4ODUtils.dataSetName, "Iteration", "Avg. Incoming Edge Weight");
//        System.out.print(minID+":");
//        System.out.println(discardFeats.get(minID));
        StringBuilder temp = new StringBuilder();
        temp.append(discardFeats.get(maxID));
        temp.append(discardFeats.size()+1);
//        System.out.println(temp);
//        System.out.println(discardFeats.size());
//        System.out.println(discardFeats.set(minID, temp.toString()));
        discardFeats.set(maxID, temp.toString());
        System.out.println(discardFeats);
        return discardFeats.get(maxID);
    }
   
    
    /**
     * to calculate the outlierness of each feature value based on the extent the value frequent deviating from the mode frequency
     * @param cpList the list of coupled centroids: each centroid contains the co-occurrence frequency of each value with other values
     * @param dataSize the number of instances in the data set
     */
    public double[] calcIntraFeatureWeight(ArrayList<CoupledValueCentroid> cpList, int dataSize) {
        int dim = cpList.size();
        double [] fo = new double[dim];
        double [] mFreq = new double[dim];
//        System.out.println();
        for(int i = 0; i < cpList.size(); i++) {
            CoupledValueCentroid cp = cpList.get(i);
            int len = cp.getCenList().size();
            double maxFreq = 0;
            for(int j = 0; j < len; j++) {
//            	  KYB: cp.getCenList() 列数_值域 0_0 0_1 0_2 0_3 0_4 0_5 1_0 1_1 1_2 1_3
                ValueCentroid cen = cp.getCenList().get(j);
//                KYB:globalFreq(列数,值)=出现次数
                double globalFreq = cen.globalFreq(i, j);
//                System.out.print("globalfreq("+i+","+j+"):"+globalFreq+", ");
                if(globalFreq > maxFreq)
                    maxFreq = globalFreq;
            }
            mFreq[i] = maxFreq;
            fo[i] = 0;
            
        }
//        System.out.println();
//        KYB: fo[]=0
        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;
        for(int i = 0; i < cpList.size(); i++) {
            CoupledValueCentroid cp = cpList.get(i);
            int len = cp.getCenList().size();
            int count = 0;
            for(int j = 0; j < len; j++) {
                ValueCentroid cen = cp.getCenList().get(j);
                double globalFreq = cen.globalFreq(i, j);
                if(globalFreq == 0) {
                    continue;
                }
                double intra;
//                特征内部离群分数
                intra = (Math.abs(globalFreq-mFreq[i])+1.0/dataSize)/(mFreq[i]); //mode absolute difference based. '1.0/dataSize' is used to avoid zero outlierness
//                System.out.println(intra);
                cen.setIntraOD(intra);
                fo[i] = fo[i] + intra;
                count++;
            }
            if(fo[i]>max)
                max = fo[i];
            if(fo[i]<min)
                min = fo[i];
        }
        double interval = max - min;
        // KYB:normalize
//        System.out.print("normalized fo: ");
        for(int i = 0; i < dim; i++) {
            fo[i] = (fo[i] - min)/interval;
//            System.out.print(fo[i]+", ");
        }
//        System.out.println();
        return fo;
    }
    /**
     * to generate the adjacent matrix for the FEATURE graph
     * @param cpList the list of coupled centroids: each centroid contains the co-occurrence frequency of each value with other values
     */
    public void featureAdjacentMatrix(ArrayList<CoupledValueCentroid> cpList,double[] fo) {
        int dim = cpList.size();
        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;
        for(int k = 0; k < dim; k++) {
            ArrayList<Double> col = new ArrayList<Double>();
            double fWgt=0;
//            int count = 0;
            for(int i = 0; i < cpList.size(); i++) {
                if(i==k) {
//                    col.add(fo[i]);
//                    fWgt += fo[i];
                    col.add(0.0);
                    continue;
                }
                double tmp = 0;
                CoupledValueCentroid cp = cpList.get(i);
                for(int j = 0; j < cp.getCenList().size(); j++) {
                    ValueCentroid cen = cp.getCenList().get(j);
                    if(cen.globalFreq(i, j) == 0)  //skip feature values that have no occurence
                        continue;
                    
                    FeatureInfo ai = cen.getAttrList().get(k);
                    FeatureInfo gai = cp.getGlobalCentroid().getAttrList().get(k);
                    int len = ai.NumofValue();
                    for(int l=0; l < len; l++) {
                        if (k==cen.getOrgFeat() && gai.value(l) != 0) { //skip zero-appearance values
                            continue;
                        }/**/
                        double freq = ai.value(l);
                        double gFreq = gai.value(l);
                        double cenFreq = cen.globalFreq(i, j);
                        if(cenFreq != 0 && gFreq != 0) { //skip zero-appearance values
                            double w = cen.getIntraOD() * cpList.get(k).getCenList().get(l).getIntraOD() * (freq*1.0/cenFreq) 
                                    + cen.getIntraOD() * cpList.get(k).getCenList().get(l).getIntraOD() * (freq*1.0/gFreq);                            
                            if(w > 0) {
//                                wlist.add(w);
                                tmp += w;
                            }
                        }
                    }
                }
                col.add(tmp);
                if(tmp>max)
                    max = tmp;
                if(tmp<min)
                    min = tmp;
                fWgt += tmp;
            }
            fMatrix.add(col);
            featStatus.add(1);
            featIndice.add(k+1);
        }
//        System.out.println("fMatrix:");
//        System.out.println(fMatrix);
        double interval = max - min;
        ArrayList<Double> tmp = new ArrayList<Double>();
        int len = fMatrix.size();
        for(int i = 0; i < len; i++) {
            double d = 0;
            ArrayList<Double> col = fMatrix.get(i);
            for(int j = 0; j < len; j++) {
                if(i == j){
                    col.add(fo[j]);
                    d += fo[j];
//                    System.out.println(fo[j]+",");
                }
                double w = (col.get(j)-min)/interval;
                col.set(j, w);
                d += w;
            }
            tmp.add(d);
//            System.out.println("col: "+col);
        }
        featIndWgts = tmp;
    }
    
    
    /**
     * to search for the densest subgraph in indirected graphs by using Charikar's greedy method presented in the paper below
     * @incollection{charikar2000greedy,
     * title={Greedy approximation algorithms for finding dense components in a graph},
     *   author={Charikar, Moses},
     *   booktitle={Approximation Algorithms for Combinatorial Optimization},
     *   pages={84--95},
     *   year={2000},
     *   publisher={Springer}
     * }
     * @param discardFeats the list to store non-relevant feature ids
     * @return the density array that records all densities of all the subgraphs
     */
    public double[] charikarGreedySearchforFeatGraph(ArrayList<String> discardFeats, double[][] dataMatrix) {
//    	System.out.println(dataMatrix.length);
    	ArrayList<Double> S = new ArrayList<Double>();
        int len = featIndWgts.size();
        int count = len;
        int id = 0;
        double[] den = new double[count];
        StringBuilder sb = new StringBuilder();
//        System.out.println(count);
        System.out.print("Subgraph densities:");
        while(count > 0) {
            double density = 0;
//            System.out.println();
//            System.out.println("当前feaIndWgts:"+featIndWgts);
            density = computeDensity(featIndWgts,count);
            den[id++] = density;
            discardFeats.add(sb.toString());
            // search
            double max = -Double.MAX_VALUE;
            int mid = -1;
            // relief抽取次数
            int reliefM = 2000;
            if(dataMatrix.length > 10000) {
            	reliefM = (int) (dataMatrix.length * 0.2);
            }
            double[][] tMatrix = dataMatrix;
            for(int i = 0; i < featIndWgts.size(); i++) {
            	double w = featIndWgts.get(i);
//            	System.out.println("w:"+w);
            	if(w > max) {
            		max = w;
            		mid = i;
            	}
            }
//            System.out.println("count="+count+",mid="+mid);
            double[] weight = relief(tMatrix,tMatrix[0].length, reliefM);
            
            S.add(featIndWgts.get(mid));
            removeOneFeature(mid);
            sb.append(featIndice.remove(mid)+",");
            count--;
            
            if(count > 1) {
//            	density = computeDensity(featIndWgts,count);
//                den[id++] = density;
//                discardFeats.add(sb.toString());
                
            	for(int n = mid; n < weight.length-1; n++) {
//                	System.out.print(new DecimalFormat("#0.0000").format(weight[n])+",");
                	weight[n] = weight[n+1];
                }
                tMatrix = delCol(tMatrix, mid);
                double[] rmWeight = relief(tMatrix,tMatrix[0].length, reliefM);
                int wId = findCompleFeature(weight, rmWeight, count);
                
                if(wId != -1) {
                	density = computeDensity(featIndWgts,count);
                	// KYB
                	ArrayList<Double> tempS = S;
                	tempS.add(featIndWgts.get(wId));
                	if(computeDensity(tempS, tempS.size()) > computeDensity(S, S.size())) {
//                		den[id++] = density;
                        sb.append(featIndice.remove(wId)+",");
                        S.add(featIndWgts.get(wId));
                	}
                    den[id++] = density;
                    discardFeats.add(sb.toString());
                	removeOneFeature(wId);
//                    sb.append(featIndice.remove(wId)+",");
//                    System.out.println("count="+count+",wId="+wId);
                    count--;
                	}
            
            }
        }
//        System.out.println(den);
        return den;
    }
    
    public int findCompleFeature(double[] weight, double[] rmWeight, int count){
    	int id = -1;
    	double[] deWeight = new double[count-1];
    	double maxdecre = -Double.MAX_VALUE;
//    	System.out.print("deWeight:");
    	for(int i = 0; i < count-1; i++) {
    		deWeight[i] = weight[i] - rmWeight[i];
    		if(deWeight[i] > maxdecre) {
    			maxdecre = deWeight[i];
    			if(maxdecre > 0) {
    				id = i;
    			}
    		}
//    		System.out.println(deWeight[i]+",");
    	}
//    	System.out.println("maxdecre:"+maxdecre+",");
    	return id;
    }
    
    public double[][] delCol(double[][] matrix, int delColnum){
    	double[][] nMatrix = new double[matrix.length][matrix[0].length-1];
    	for(int i = 0; i < matrix.length; i++) {
    		for(int j = delColnum; j < matrix[0].length-1; j++) {
    			matrix[i][j] = matrix[i][j+1];
    		}
    		for(int k = 0; k < nMatrix[0].length; k++) {
        		nMatrix[i][k] = matrix[i][k];
        	}
    	}
    	return nMatrix;
    }
    
    // Relief计算权重
    public double[] relief(double[][] matrix, int n_vars, int m) {
    	int length = matrix.length;
    	int width = matrix[0].length;
    	// 权重置0
    	double[] weight = new double[n_vars];
    	for(int i = 0; i < n_vars; i++) {
    		weight[i] = 0.0;
    	}
    	// 特征的最大值和最小值
    	double[] max = new double[n_vars];
    	double[] min = new double[n_vars];
    	for(int i = 0; i < width; i++) {
    		for(int j = 0; j < length; j++) {
    			double d = matrix[j][i];
    			if(d > max[i]) {
    				max[i] = d;
    			}
    			if(d < min[i]) {
    				min[i] = d;
    			}
    		}
    	}
    	// 随机抽样
    	for(int i = 0; i < m; i++) {
    		Random random = new Random();
    		int R_index = random.nextInt(length);
    		double[] R = new double[width];
    		for(int index = 0; index < width; index++) {
    			R[index] = matrix[R_index][index];
    		}
    		double H_value = 0.0;
    		double M_value = 0.0;
    		int H_row = 0;
    		int M_row = 0;
    		double distince = 0.0;
    		for(int len = 0; len < length; len++) {
    			if(len != R_index) {
    				for(int wid = 0; wid < width; wid++) {
//    					distince += Math.pow(R[wid]-matrix[len][wid], 2);
//    					System.out.println("matrix[len][wid], R[wid]"+matrix[len][wid]+","+ R[wid]);
    					if(matrix[len][wid] != R[wid]) {
    						distince += Math.pow(R[wid]-matrix[len][wid], 2);
    					}
    				}
    				distince = Math.sqrt(distince);
    				if(matrix[len][width-1] == matrix[len][width-1]) {
    					if(len == 0) {
        					H_value = distince;
        				}
    					if(distince < H_value) {
    						H_value = distince;
    						H_row = len;
    					}
    				}
    				if(matrix[len][width-1] != matrix[len][width-1]) {
    					if(len == 0) {
        					M_value = distince;
        				}
    					if(distince < H_value) {
    						M_value = distince;
    						M_row = len;
    					}
    				}
    			}
    		}
    		
    		double[] H = new double[width];
    		for(int index = 0; index < width; index++) {
    			H[index] = matrix[H_row][index];
    		}
    		double[] M = new double[width];
    		for(int index = 0; index < width; index++) {
    			M[index] = matrix[M_row][index];
    		}
    		for(int j = 0; j < n_vars; j++) {
				weight[j] = weight[j]-(Math.abs(R[j]-H[j])/(max[j]-min[j]))/m + (Math.abs(R[j]-M[j])/(max[j]-min[j]))/m; 
//				System.out.print(new DecimalFormat("#0.0000").format(weight[i])+",");
			}
    	}
    	
//    	for(int i = 0; i < width; i++) {
//    		System.out.print(new DecimalFormat("#0.0000").format(weight[i])+",");
//    	}
//    	System.out.println();
    	return weight;
	}
    
    /**
     * to remove one feature from the feature candidates
     * @param fid the id of the feature to be removed
     */    
    public void removeOneFeature(int fid) {
//    	System.out.println();
//    	System.out.println("fMatrix:"+fMatrix);
        fMatrix.remove(fid);
        featIndWgts.remove(fid); //virtually remove the feature
//            double fWgt = featIndWgts.get(fid);
        for(int k = 0; k < fMatrix.size(); k++) {
            ArrayList<Double> col = fMatrix.get(k);
            double fWgt = featIndWgts.get(k);
            
            double w = col.remove(fid);
//                double w = col.get(fid);
//            System.out.println("fWgt="+fWgt+",w="+w);
            featIndWgts.set(k, fWgt-w);
        }
    }
    
    /**
     * to compute the subgraph density using feature level array-list <code>featIndWgts</code>, i.e., average weight per node
     * @param edgeWeights the total incoming edge weights of individual feature values
     * @param featNum the number of features left
     * @param sb to store the non-relevant feature ids
     * @return the subgraph density
     */
    public double computeDensity(ArrayList<Double> edgeWeights, int featNum) {
        double density = 0;
        int len = edgeWeights.size();
        for(int i = 0; i < len; i++ ) {
            double w = edgeWeights.get(i);
            density += w;
        }
        density = density / featNum;
//        System.out.print("当前子图密度：");
        System.out.print((new DecimalFormat("#0.0000")).format(density)+",");
//        recordDen.add(density);
        return density;
    }
    
    /**
     * to conduct actual feature selection and generate a new data set given a list of non-relevant feature indices
     * @param data the data set
     * @param str the non-relevant feature indices
     * @param path the path for storing the new data set
     * @param name  name of the data
     */
    public void featureSelection(Instances data, String str, String path, String name){
        
        BufferedWriter writer = null;
        Remove remove = new Remove();
//        System.out.println(str);
        remove.setAttributeIndices(str);
//      反向选择
        remove.setInvertSelection(true);
        try {
            remove.setInputFormat(data);
            Instances newData = Filter.useFilter(data, remove);
//            Instances newData = data;
//            File newDir = new File(path+"\\"+name);
//            if(!newDir.exists())
//                newDir.mkdir();
            // System.out.print(String.format(fm, count)+",");
            int num = 0;
            if(str.split(",").length>0)
                num = str.split(",").length - 1;
            else
                num = data.numAttributes()-1;
//            System.out.println(str.split(",").length);
//            writer = new BufferedWriter(new FileWriter(path+"FS_"+name+"_"+(data.numAttributes()-1)+"to"+num+".arff"));
//            writer.write(newData.toString());
//            writer.flush();
//            writer.close();
            remove.setInputFormat(data);
        } catch (Exception ex) {
            Logger.getLogger(SubgraphDiscovery.class.getName()).log(Level.SEVERE, null, ex);
        }
        
    }
    
    
}
