/*
 * Example script to output the number of objects of different types,
 * identified using two different methods.
 */

import qupath.lib.io.PathIO;
import groovy.io.FileType

public void exportQpdataAsJSON(final File srcfile) {
    String savename = srcfile.getName();
    savename = savename.substring(0, savename.lastIndexOf('.')) + '.json';
    File savefile = new File(srcfile.getParentFile(), savename)
    
    String result = '[ \n'
    hierarchy = PathIO.readHierarchy(srcfile)
    for (annotation in hierarchy.getAnnotationObjects()){
        roi = annotation.getROI()
        def pathClass = annotation.getPathClass()
        result += String.format('["%s" , [', pathClass)
        
        for (p in roi.getPolygonPoints()) {
            result += String.format("[%.2f, %.2f],", p.getX(), p.getY()) 
        }
        
        result = result.substring(0, result.length() - 1); //remove the last comma
        result += "]],\n"
    }
    result = result.substring(0, result.length() - 2); //remove the last comma
    result += '\n]'
    
    fileWriter = new FileWriter(savefile)
    fileWriter.write(result)
    fileWriter.flush()
    fileWriter.close()
    
    print("Exported: " + savefile)
    }

println "Grovy version: " + GroovySystem.version
println "Java version: " + System.getProperty("java.version")

//File srcfile = new File("/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/eosinophils_regions/A1005474.qpdata");
//File srcfile = new File("/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/eosinophils_regions/A1005474.json");
//exportQpdataAsJSON(srcfile)

def rootdir = new File("/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/eosinophils_regions/");
rootdir.traverse(type: FileType.FILES, maxDepth: 0) { 
    if (it.name.endsWith('.qpdata')) {
        exportQpdataAsJSON(it)
    }
};

