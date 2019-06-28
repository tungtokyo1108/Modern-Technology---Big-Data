/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-data-lake-store-java/blob/master/src/main/java/com/microsoft/azure/datalake/store/Operation.java
 *
 *  Created on: June 27, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.datalake.store.mirror;

/**
 * The WebHDFS methods, and their associated properties
 */
enum Operation {
    OPEN ("OPEN", "GET", C.requiresBodyFalse, C.returnsBodyTrue, C.enforceMimeTypeJsonFalse, C.webHdfs),
    GETFILESTATUS ("GETFILESTATUS", "GET", C.requiresBodyFalse, C.returnsBodyTrue, C.enforceMimeTypeJsonFalse, C.webHdfs),
    MSGETFILESSTATUS ("MSGETFILESSTATUS", "GET", C.requiresBodyFalse, C.returnsBodyTrue, C.enforceMimeTypeJsonFalse, C.webHdfs),
    LISTSTATUS ("LISTSTATUS", "GET", C.requiresBodyFalse, C.returnsBodyTrue, C.enforceMimeTypeJsonFalse, C.webHdfs),
    MSLISTSTATUS ("MSLISTSTATUS", "GET", C.requiresBodyFalse, C.returnsBodyTrue, C.enforceMimeTypeJsonFalse, C.webHdfs),
    GETCONTENTSUMMARY ("GETCONTENTSUMMARY", "GET", C.requiresBodyFalse, C.returnsBodyTrue, C.enforceMimeTypeJsonFalse, C.webHdfs),
    GETFILECHECKSUM ("GETFILECHECKSUM", "GET", C.requiresBodyFalse, C.returnsBodyTrue, C.enforceMimeTypeJsonFalse, C.webHdfs),
    GETACLSTATUS ("GETACLSTATUS", "GET", C.requiresBodyFalse, C.returnsBodyTrue, C.enforceMimeTypeJsonFalse, C.webHdfs), 
    MSGETACLSTATUS ("MSGETACLSTATUS", "GET", C.requiresBodyFalse, C.returnsBodyTrue, C.enforceMimeTypeJsonFalse, C.webHdfs),
    CHECKACCESS ("CHECKACCESS", "GET", C.requiresBodyFalse, C.returnsBodyFalse, C.enforceMimeTypeJsonFalse, C.webHdfs),
    CREATE ("CREATE", "PUT", C.requiresBodyTrue, C.returnsBodyFalse, C.enforceMimeTypeJsonFalse, C.webHdfs),
    MKDIRS ("MKDIRS", "PUT", C.requiresBodyFalse, C.returnsBodyTrue, C.enforceMimeTypeJsonFalse, C.webHdfs),
    RENAME ("RENAME", "PUT", C.requiresBodyFalse, C.returnsBodyTrue, C.enforceMimeTypeJsonFalse, C.webHdfs), 
    SETOWNER ("SETOWNER", "PUT", C.requiresBodyFalse, C.returnsBodyFalse, C.enforceMimeTypeJsonFalse, C.webHdfs),
    SETPERMISSION ("SETPERMISSION", "PUT", C.requiresBodyFalse, C.returnsBodyFalse, C.enforceMimeTypeJsonFalse, C.webHdfs),
    SETTIMES ("SETTIMES", "PUT", C.requiresBodyFalse, C.returnsBodyFalse, C.enforceMimeTypeJsonFalse, C.webHdfs),
    MODIFYACLENTRIES ("MODIFYACLENTRIES", "PUT", C.requiresBodyFalse, C.returnsBodyFalse, C.enforceMimeTypeJsonFalse, C.webHdfs),
    REMOVEACLENTRIES ("REMOVEACLENTRIES", "PUT", C.requiresBodyFalse, C.returnsBodyFalse, C.enforceMimeTypeJsonFalse, C.webHdfs),
    REMOVEDEFAULTACL ("REMOVEDEFAULTACL", "PUT", C.requiresBodyFalse, C.returnsBodyFalse, C.enforceMimeTypeJsonFalse, C.webHdfs),
    REMOVEACL ("REMOVEACL", "PUT", C.requiresBodyFalse, C.returnsBodyFalse, C.enforceMimeTypeJsonFalse, C.webHdfs),
    SETACL ("SETACL", "PUT", C.requiresBodyFalse, C.returnsBodyFalse, C.enforceMimeTypeJsonFalse, C.webHdfs),
    CREATENONRECURSIVE ("CREATENONRECURSIVE", "PUT", C.requiresBodyFalse, C.returnsBodyFalse, C.enforceMimeTypeJsonFalse, C.webHdfs),
    APPEND ("APPEND", "POST", C.requiresBodyTrue, C.returnsBodyFalse, C.enforceMimeTypeJsonFalse, C.webHdfs),
    CONCAT ("CONCAT", "POST", C.requiresBodyFalse, C.returnsBodyFalse, C.enforceMimeTypeJsonFalse, C.webHdfs),
    MSCONCAT ("MSCONCAT", "POST", C.requiresBodyTrue, C.returnsBodyFalse, C.enforceMimeTypeJsonTrue, C.webHdfs),
    DELETE ("DELETE", "DELETE", C.requiresBodyFalse, C.returnsBodyTrue, C.enforceMimeTypeJsonFalse, C.webHdfs),
    CONCURRENTAPPEND ("CONCURRENTAPPEND", "POST", C.requiresBodyTrue, C.returnsBodyFalse, C.enforceMimeTypeJsonFalse, C.webHdfsExt),
    SETEXPIRY ("SETEXPIRY", "PUT", C.requiresBodyFalse, C.returnsBodyFalse, C.enforceMimeTypeJsonFalse, C.webHdfsExt),
    GETFILEINFO ("GETFILEINFO", "GET", C.requiresBodyFalse, C.returnsBodyFalse, C.enforceMimeTypeJsonFalse, C.webHdfsExt);


    String name;
    String method;
    boolean requiresBody;
    boolean returnsBody;
    boolean enforceMimeTypeJson;
    String namespace; 

    Operation(String name, String method, boolean requiresBody, boolean returnsBody, boolean enforceMimeTypeJson, String namespace)
    {
        this.name = name;
        this.method = method;
        this.requiresBody = requiresBody;
        this.returnsBody = returnsBody;
        this.enforceMimeTypeJson = enforceMimeTypeJson;
        this.namespace = namespace;
    }

    private static class C {
        static final boolean requiresBodyTrue = true;
        static final boolean requiresBodyFalse = true;
        static final boolean returnsBodyTrue = true;
        static final boolean returnsBodyFalse = false;
        static final boolean enforceMimeTypeJsonFalse = false;
        static final boolean enforceMimeTypeJsonTrue = true;
        static final String webHdfs = "/webhfs/v1";
        static final String webHdfsExt = "/WebHdfsExt";
    }
}