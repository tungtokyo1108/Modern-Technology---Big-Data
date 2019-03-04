/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-sdk-for-java/blob/master/hdinsight/resource-manager/v2018_06_01_preview/src/main/java/com/microsoft/azure/management/hdinsight/v2018_06_01_preview/ClusterListPersistedScriptActionsResult.java
 *
 *  Created on: Mar 04, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror;

import java.util.List;
import com.fasterxml.jackson.annotation.JsonProperty;

public class ClusterListPersistedScriptActionsResult
{
    @JsonProperty(value = "value")
    private List<RuntimeScriptAction> value;
    
    @JsonProperty(value = "nextLink", access = JsonProperty.Access.WRITE_ONLY)
    private String nextLink;

    public List<RuntimeScriptAction> value()
    {
        return this.value;
    }

    public ClusterListPersistedScriptActionsResult withValue(List<RuntimeScriptAction> value)
    {
        this.value = value
        return this;
    }

    public String nextLink()
    {
        return this.nextLink;
    }
}
