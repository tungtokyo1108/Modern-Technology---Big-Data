/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-sdk-for-java/blob/master/hdinsight/resource-manager/v2018_06_01_preview/src/main/java/com/microsoft/azure/management/hdinsight/v2018_06_01_preview/ClusterDefinition.java
 *
 *  Created on: Mar 04, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror;

import java.util.Map;
import com.fasterxml.jackson.annotation.JsonProperty;

public class ClusterDefinition 
{
    @JsonProperty(value = "blueprint")
    private String blueprint;

    @JsonProperty(value = "kind")
    private String kind;

    @JsonProperty(value = "componentVersion")
    private Map<String, String> componentVersion;

    @JsonProperty(value = "configurations")
    private Object configurations;

    public String blueprint() 
    {
        return this.blueprint;
    }

    public ClusterDefinition withBlueprint(String blueprint)
    {
        this.blueprint = blueprint;
        return this;
    }

    public String kind() 
    {
        return this.kind;
    }

    public ClusterDefinition withKind(String kind)
    {
        this.kind = kind;
        return this;
    }

    public Map<String, String> componentVersion() 
    {
        return this.componentVersion;
    }

    public ClusterDefinition withComponentVersion(Map<String, String> componentVersion)
    {
        this.componentVersion = componentVersion;
        return this;
    }

    public Object configurations()
    {
        return this.configurations;
    }

    public ClusterDefinition withConfigurations(Object configurations)
    {
        this.configurations = configurations;
        return this;
    }
}
