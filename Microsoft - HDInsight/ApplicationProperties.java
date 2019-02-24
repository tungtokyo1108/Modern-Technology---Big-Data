/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-sdk-for-java/blob/master/hdinsight/resource-manager/v2018_06_01_preview/src/main/java/com/microsoft/azure/management/hdinsight/v2018_06_01_preview/ApplicationProperties.java
 *
 *  Created on: Feb 24, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror;

import java.util.List;
import com.fasterxml.jackson.annotation.JsonProperty;

public class ApplicationProperties 
{
    @JsonProperty(value = "computeProfile")
    private ComputeProfile computeProfile;
    
    @JsonProperty(value = "installScriptActions")
    private List<RuntimeScripAction> installScripActions;

    @JsonProperty(value = "uninstallScriptActions")
    private List<RuntimeScripAction> uninstallScriptActions;

    @JsonProperty(value = "httpsEndpoints")
    private List<ApplicationGetHttpsEndpoint> httpsEndpoints;

    @JsonProperty(value = "sshEndpoints")
    private List<ApplicationGetEndpoint> sshEndpoints;

    @JsonProperty(value = "provisioningState", access = JsonProperty.Access.WRITE_ONLY)
    private String provisioningState;

    @JsonProperty(value = "applicationType")
    private String applicationType;

    @JsonProperty(value = "applicationState", access = JsonProperty.Access.WRITE_ONLY)
    private String applicationState;

    @JsonProperty(value = "errors")
    private List<Errors> errors;

    @JsonProperty(value = "createDate", access = JsonProperty.Access.WRITE_ONLY)
    private String createDate;

    @JsonProperty(value = "marketplacedentifier", access = JsonProperty.Access.WRITE_ONLY)
    private String marketplacedentifier;

    public ComputeProfile computeProfile() 
    {
        return this.computeProfile;
    }

    public ApplicationProperties withComputeProfile(ComputeProfile computeProfile)
    {
        this.computeProfile = computeProfile;
        return this;
    }

    public List<RuntimeScriptAction> installScriptActions()
    {
        return this.installScripActions;
    }

    public ApplicationProperties withInstallScriptActions(List<RuntimeScriptAction> installScriptActions)
    {
        this.installScriptActions = installScriptActions;
        return this;
    }

    public List<RuntimeScriptAction> uninstallScriptActions()
    {
        return this.uninstallScriptActions;
    }

    public ApplicationProperties withUninstallScriptActions(List<RuntimeScriptAction> uninstallScriptActions)
    {
        this.uninstallScriptActions = uninstallScriptActions;
        return this;
    }

    public List<ApplicationGetHttpsEndpoint> httpsEndpoints()
    {
        return this.httpsEndpoints;
    }

    public ApplicationProperties withHttpsEndpoints(List<ApplicationGetHttpsEndpoint> httpsEndpoints)
    {
        this.httpsEndpoints = httpsEndpoints;
        return this;
    }

    public List<ApplicationGetEndpoint> sshEndpoints()
    {
        return this.sshEndpoints;
    }

    public ApplicationProperties withSshEndpoints(List<ApplicationGetEndpoint> sshEndpoints)
    {
        this.sshEndpoints = sshEndpoints;
        return this;
    }

    public String provisioningState() 
    {
        return this.provisioningState;
    }

    public String applicationType()
    {
        return this.applicationType;
    }

    public ApplicationProperties withApplicationType(String applicationType)
    {
        this.applicationType = applicationType;
        return this;
    }

    public String applicationState()
    {
        return this.applicationState;
    }

    public List<Errors> errors()
    {
        return this.errors;
    }

    public ApplicationProperties withErrors(List<Errors> errors)
    {
        this.errors = errors;
        return this;
    }

    public String createDate()
    {
        return this.createdDate;
    }

    public String marketplacedentifier()
    {
        return this.marketplacedentifier;
    }
}
