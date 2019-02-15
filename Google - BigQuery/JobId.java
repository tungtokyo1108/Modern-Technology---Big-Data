/**
 * Google - Big Data Technology
 * https://github.com/googleapis/google-cloud-java/blob/master/google-cloud-clients/google-cloud-bigquery/src/main/java/com/google/cloud/bigquery/JobId.java
 *
 *  Created on: Feb 14, 2019
 *  Student (MIG Virtual Developer): Tung Dang
 */

package com.google.cloud.bigquery.mirror;

import static com.google.common.base.Preconditions.checkNotNull;
import com.google.api.services.bigquery.model.JobReference;
import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableBiMap.Builder;

import java.io.Serializable;
import java.util.UUID;
import javax.annotation.Nullable;

@AutoValue
public abstract class JobId implements Serializable 
{
    private static final long serialVersionUID = 1225914835379688977L;
    JobId(){}
    @Nullable
    public abstract String getProject();

    @Nullable
    public abstract String getJob();

    @Nullable
    public abstract String getLocation();
    public abstract Builder toBuilder();
    public static Builder newBuilder()
    {
        return new AutoValue_JobId.Builder();
    }

    @AutoValue.Builder
    public abstract static class Builder 
    {
        public abstract Builder setProject(String project);
        public abstract Builder setJob(String job);
        public Builder setRandomJob()
        {
            return setJob(UUID.randomUUID().toString());
        }
        public abstract Builder setLocation(String location);
        public abstract JobId build();
    }

    public static JobId of(String project, String job)
    {
        return newBuilder().setProject(checkNotNull(project)).setJob(checkNotNull(job)).build();
    }

    public static JobId of(String job)
    {
        return newBuilder().setJob(checkNotNull(job)).build();
    }

    public static JobId of()
    {
        return newBuilder().setRandomJob().build();
    }

    JobId setProjectId(String projectId)
    {
        return getProject() != null ? this : toBuilder().setProject(projectId).build();
    }

    JobId setLocation(String location)
    {
        return getLocation() != null ? this : toBuilder().setLocation(location).build();
    }

    JobReference toPb()
    {
        return new JobReference()
            .setProjectId(getProject())
            .setJobId(getJob())
            .setLocation(getLocation());
    }

    static JobId fromPb(JobReference jobRef)
    {
        return newBuilder() 
            .setProject(jobRef.getProjectId())
            .setJob(jobRef.getJobId())
            .setLocation(jobRef.getLocation())
            .build();
    }
}
