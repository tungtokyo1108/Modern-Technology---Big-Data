/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-sdk-for-java/blob/master/hdinsight/resource-manager/v2018_06_01_preview/src/main/java/com/microsoft/azure/management/hdinsight/v2018_06_01_preview/Clusters.java
 *
 *  Created on: Mar 06, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror;

import com.microsoft.azure.arm.collection.SupportsCreating;
import com.microsoft.azure.arm.resources.collection.SupportsDeletingByResourceGroup;
import com.microsoft.azure.arm.resources.collection.SupportsBatchDeletion;
import com.microsoft.azure.arm.resources.collection.SupportsGettingByResourceGroup;
import rx.Observable;
import com.microsoft.azure.arm.resources.collection.SupportsListingByResourceGroup;
import com.microsoft.azure.collection.SupportsListing;
import rx.Completable;
import com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror.ClusterInner;
import com.microsoft.azure.arm.model.HasInner;

public interface Clusters extends SupportsCreating<Clusters.DefinitionStages.Blank>, 
        SupportsDeletingByResourceGroup, SupportsBatchDeletion, SupportsGettingByResourceGroup<Cluster>,
        SupportsListingByResourceGroup<Cluster>, SupportsListing<Cluster>, HasInner<ClusterInner>
{
    Completable rotateDiskEncryptionKeyAsync(String resourceGroupName, String clusterName, ClusterDiskEncryptionParameters parameters);

    Completable executeScriptActionsAsync(String resourceGroupName, String clusterName, ExecuteScriptActionParameters parameters);

    Completable resizeAsync(String resourceGroupName, String clusterName);
}
