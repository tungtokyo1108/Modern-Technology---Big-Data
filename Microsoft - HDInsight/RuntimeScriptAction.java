/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-sdk-for-java/blob/master/hdinsight/resource-manager/v2018_06_01_preview/src/main/java/com/microsoft/azure/management/hdinsight/v2018_06_01_preview/RuntimeScriptAction.java
 *
 *  Created on: Mar 06, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror;

import java.util.List;
import com.fasterxml.jackson.annotation.JsonProperty;

public class RuntimeScriptAction 
{
    @JsonProperty(value = "name", required = true)
    private String name;

    @JsonProperty(value = "uri", required = true)
    private String uri;

    @JsonProperty(value = "parameters")
    private String parameters;

    @JsonProperty(value = "roles", required = true)
    private List<String> roles;

    @JsonProperty(value = "applicationName", access = JsonProperty.Access.WRITE_ONLY)
    private String applicationName;

    public String name()
    {
        return this.name;
    }

    public RuntimeScriptAction withName(String name)
    {
        this.name = name;
        return this;
    }

    public String uri()
    {
        return this.uri;
    }

    public RuntimeScriptAction withUri(String uri)
    {
        this.uri = uri;
        return this;
    }

    public String parameters()
    {
        return this.parameters;
    }

    public RuntimeScriptAction withParameters(String parameters) 
    {
        this.parameters = parameters;
        return this;
    }

    public List<String> roles()
    {
        return this.roles;
    }

    public RuntimeScriptAction withRoles(List<String> roles)
    {
        this.roles = roles;
        return this;
    }

    public String applicationName()
    {
        return this.applicationName;
    }
}
