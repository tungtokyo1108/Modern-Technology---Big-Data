/**
 * Google - Big Data Technology
 * https://github.com/googleapis/google-cloud-java/blob/master/google-cloud-clients/google-cloud-bigquery/src/main/java/com/google/cloud/bigquery/DatasetId.java
 *
 *  Created on: Feb 13, 2019
 *  Student (MIG Virtual Developer): Tung Dang
 */

package com.google.cloud.bigquery.mirror;

import static com.google.common.base.Preconditions.checkNotNull;
import com.google.api.services.bigquery.model.DatasetReference;
import java.io.Serializable;
import java.util.Objects;

public final class DatasetId implements Serializable 
{
    private static final long serialVersionUID = -6186254820908152300L;

    private final String project;
    private final String dataset;

    public String getProject() {
        return project;
    }

    public String getDataset() {
        return dataset;
    }

    private DatasetId (String project, String dataset)
    {
        this.dataset = dataset;
        this.project = project;
    }

    /* Creates a dataset identity */
    public static DatasetId of(String project, String dataset)
    {
        return new DatasetId(checkNotNull(project), checkNotNull(dataset));
    }
    public static DatasetId of(String dataset)
    {
        return new DatasetId(null, checkNotNull(dataset));
    }

    @Override
    public boolean equals(Object obj) 
    {
        return obj == this 
            || obj instanceof DatasetId && Objects.equals(toPb(), ((DatasetId) obj).toPb());
    }

    @Override 
    public int hashCode() 
    {
        return Objects.hash(project, dataset);
    }

    @Override
    public String toString()
    {
        return toPb().toString();
    }

    DatasetId setProjectId(String projectId)
    {
        return getProject() != null ? this : DatasetId.of(projectId, getDataset());
    }

    DatasetReference toPb()
    {
        return new DatasetReference().setProjectId(project).setDatasetId(dataset);
    }

    static DatasetId fromPb(DatasetReference datasetRef)
    {
        return new DatasetId(datasetRef.getProjectId(), datasetRef.getDatasetId());
    }
}
