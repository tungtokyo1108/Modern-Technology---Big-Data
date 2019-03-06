/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-sdk-for-java/blob/master/hdinsight/resource-manager/v2018_06_01_preview/src/main/java/com/microsoft/azure/management/hdinsight/v2018_06_01_preview/ExecuteScriptActionParameters.java
 *
 *  Created on: Mar 06, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror;

import java.util.List;
import com.fasterxml.jackson.annotation.JsonProperty;

public class ExecuteScriptActionParameters 
{
    @JsonProperty(value = "scriptActions")
    private List<RuntimeScriptAction> scriptActions;

    @JsonProperty(value = "persistOnSuccess", required = true)
    private boolean persistOnSuccess;

    public List<RuntimeScriptAction> scriptActions()
    {
        return this.scriptActions;
    }

    public ExecuteScriptActionParameters withScriptActions(List<RuntimeScriptAction> scriptActions)
    {
        this.scriptActions = scriptActions;
        return this;
    }

    public boolean persistOnSuccess()
    {
        return this.persistOnSuccess;
    }

    public ExecuteScriptActionParameters withPersistOnSuccess(boolean persistOnSuccess) 
    {
        this.persistOnSuccess = persistOnSuccess;
        return this;
    }
}
