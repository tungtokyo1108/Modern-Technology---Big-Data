/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-data-lake-store-java/blob/master/src/main/java/com/microsoft/azure/datalake/store/QueryParams.java
 *
 *  Created on: June 27, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.datalake.store.mirror;

import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.util.Hashtable;
import java.util.Map;

public class QueryParams {

    private Hashtable<String, String> params = new Hashtable<String, String>();
    Operation op = null;
    String apiVersion = null;
    String separator = "";
    String serializedString = null;

    public void add(String name, String value) {
        params.put(name, value);
        serializedString = null;
    }

    public void setOp(Operation op) {
        this.op = op;
        serializedString = null;
    }

    public void setApiVersion(String apiVersion) {
        this.apiVersion = apiVersion;
        serializedString = null;
    }

    public String serialize() {
        if (serializedString == null)
        {
            StringBuilder sb = new StringBuilder();

            if (op != null)
            {
                sb.append(separator);
                sb.append("op=");
                sb.append(op.name);
                separator = "&";
            }

            for (String name : params.keySet())
            {
                try {
                    sb.append(separator);
                    sb.append(URLEncoder.encode(name, "UTF-8"));
                    sb.append('=');
                    sb.append(URLEncoder.encode(params.get(name), "UTF-8"));
                    separator = "&";
                } catch (UnsupportedEncodingException ex) {

                }
            }

            if (apiVersion != null)
            {
                sb.append(separator);
                sb.append("api-version=");
                sb.append(apiVersion);
                separator = "&";
            }

            serializedString = sb.toString();
        }
        return serializedString;
    }
}
