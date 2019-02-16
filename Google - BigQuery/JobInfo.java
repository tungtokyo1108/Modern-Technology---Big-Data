/**
 * Google - Big Data Technology
 * https://github.com/googleapis/google-cloud-java/blob/master/google-cloud-clients/google-cloud-bigquery/src/main/java/com/google/cloud/bigquery/JobInfo.java
 *
 *  Created on: Feb 16, 2019
 *  Student (MIG Virtual Developer): Tung Dang
 */

package com.google.cloud.bigquery.mirror;

import com.google.api.services.bigquery.model.Job;
import com.google.api.services.bigquery.model.JobConfiguration;
import com.google.api.services.bigquery.model.JobStatistics;
import com.google.common.base.Function;
import com.google.common.base.MoreObjects;
import com.google.rpc.RetryInfo;

import java.awt.image.PixelGrabber;
import java.io.Serializable;
import java.util.Objects;

import javax.print.attribute.standard.JobImpressionsCompleted;

public class JobInfo implements Serializable {

    static final Function<Job, JobInfo> FROM_PB_FUNCTION = 
        new Function<Job,JobInfo>() {
            @Override
            public JobInfo apply(Job pb) {
                return JobInfo.fromPb(pb);
            }
        };

    private static final long serialVersionUID = 2740548743267670124L;

    private final String etag;
    private final String generatedId;
    private final JobId jobId;
    private final String selfLink;
    private final JobStatus status;
    private final JobStatistics statistics;
    private final String userEmail;
    private final JobConfiguration configuration;

    /* Specifies whether the job is allowed to create new tables */
    public enum CreateDisposition
    {
        CREATE_IF_NEEDED,
        CREATE_NEVER
    }

    /* Specifies the action that occurs if the destination table already exist */
    public enum WriteDisposition
    {
        WRITE_TRUNCATE,
        WRITE_APPEND,
        WRITE_EMPTY
    }

    public enum SchemaUpdateOption 
    {
        ALLOW_FIELD_ADDITION,
        ALLOW_FIELD_RELAXATION
    }

    public abstract static class Builder
    {
        abstract Builder setEtag(String etag);
        abstract Builder setGeneratedId(String generatedId);
        public abstract Builder setJobId(JobId jobId);
        abstract Builder setSelfLink(String selfLink);
        abstract Builder setStatus(JobStatus status);
        abstract Builder setStatistics(JobStatistics statistics);
        abstract Builder setUserEmail(String userEmail);
        public abstract Builder setConfiguration(JobConfiguration configuration);
        public abstract JobInfo build();
    }

    static final class BuilderImpl extends Builder
    {
        private String etag;
        private String generatedId;
        private JobId jobId;
        private String selfLink;
        private JobStatus status;
        private JobStatistics statistics;
        private String userEmail;
        private JobConfiguration configuration;

        BuilderImpl() {}

        BuilderImpl(JobInfo jobInfo)
        {
            this.etag = jobInfo.etag;
            this.generatedId = jobInfo.generatedId;
            this.jobId = jobInfo.jobId;
            this.selfLink = jobInfo.selfLink;
            this.status = jobInfo.status;
            this.statistics = jobInfo.statistics;
            this.userEmail = jobInfo.userEmail;
            this.configuration = jobInfo.configuration;
        }

        BuilderImpl(Job jobPd)
        {
            this.etag = jobPd.getEtag();
            this.generatedId = jobPd.getId();
            if (jobPd.getJobReference() != null)
            {
                this.jobId = JobId.fromPb(jobPd.getJobReference());
            }
            this.selfLink = jobPd.getSelfLink();
            if (jobPd.getStatus() != null)
            {
                this.status = JobStatus.fromPb(jobPd.getStatus());
            }
            if (jobPd.getStatistics() != null)
            {
                this.statistics = JobStatistics.fromPb(jobPd);
            }
            this.userEmail = jobPd.getUserEmail();
            if (jobPd.getConfiguration() != null)
            {
                this.configuration = JobConfiguration.fromPb(jobPd.getConfiguration());
            }
        }

        @Override
        Builder setEtag(String etag)
        {
            this.etag = etag;
            return this;
        }

        @Override
        Builder setGeneratedId(String generatedId)
        {
            this.generatedId = generatedId;
            return this;
        }

        @Override
        public Builder setJobId(JobId jobId)
        {
            this.jobId = jobId;
            return this;
        }

        @Override
        Builder setSelfLink(String selfLink)
        {
            this.selfLink = selfLink;
            return this;
        }

        @Override
        Builder setStatus(JobStatus status)
        {
            this.status = status;
            return this;
        }

        @Override
        Builder setStatistics(JobStatistics statistics)
        {
            this.statistics = statistics;
            return this;
        }

        @Override 
        Builder setUserEmail(String userEmail)
        {
            this.userEmail = userEmail;
            return this;
        }

        @Override
        public Builder setConfiguration(JobConfiguration configuration)
        {
            this.configuration = configuration;
            return this;
        }

        @Override
        public JobInfo build()
        {
            return new JobInfo(this);
        }

        JobInfo(BuilderImpl builder)
        {
            this.jobId = builder.jobId;
            this.etag = builder.etag;
            this.generatedId = builder.generatedId;
            this.selfLink = builder.selfLink;
            this.status = builder.status;
            this.statistics = builder.statistics;
            this.userEmail = builder.userEmail;
            this.configuration = builder.configuration;
        }

        public String getEtag()
        {
            return etag;
        }

        public String getGeneratedId()
        {
            return generatedId;
        }

        public JobId getJobId()
        {
            return jobId;
        }

        public String getSelfLink()
        {
            return selfLink;
        }

        public JobStatus getStatus()
        {
            return status;
        }

        @SuppressWarnings("unchecked")
        public <S extends JobStatistics> S getStatistics()
        {
            return (S) statistics;
        }

        public String getUserEmail() {
            return userEmail;
        }

        @SuppressWarnings("unchecked")
        public <C extends JobConfiguration> C getConfiguration()
        {
            return (C) configuration;
        }

        public Builder toBuilder()
        {
            return new BuilderImpl(this);
        }

        @Override
        public String toString()
        {
            return MoreObjects.toStringHelper(this)
                .add("job", jobId)
                .add("status", status)
                .add("statistics", statistics)
                .add("userEmail", userEmail)
                .add("etag", etag)
                .add("generatedId", generatedId)
                .add("selfLink", selfLink)
                .add("configuration", configuration)
                .toString();
        }

        @Override
        public int hashCode()
        {
            return Objects.hash(jobId);
        }

        @Override
        public boolean equals(Object obj)
        {
            return obj == this
                || obj != null
                    && obj.getClass().equals(JobInfo.class)
                    && Objects.equals(toPb(), ((JobInfo) obj).toPb());
        }

        JobInfo setProjectId(String projectId)
        {
            Builder builder = toBuilder();
            if (jobId != null)
            {
                builder.setJobId(jobId.setProjectId(projectId));
            }
            return builder.setConfiguration(configuration.setProjectId(projectId)).build();
        }

        Job toPb()
        {
            Job jobPb = new Job();
            jobPb.setEtag(etag);
            jobPb.setId(generatedId);
            jobPb.setSelfLink(selfLink);
            jobPb.setUserEmail(userEmail);
            if (jobId != null)
            {
                jobPb.setJobReference(jobId.toPb());
            }
            if (status != null)
            {
                jobPb.setStatistics(statistics.toPb());
            }
            if (configuration != null)
            {
                jobPb.setConfiguration(configuration.toPb());
            }
            return jobPb;
        }

        public static Builder newBuilder(JobConfiguration configuration)
        {
            return new BuilderImpl().setConfiguration(configuration);
        }

        public static JobInfo of(JobConfiguration configuration)
        {
            return newBuilder(configuration).build();
        }

        public static JobInfo of(JobId jobId, JobConfiguration configuration)
        {
            return newBuilder(configuration).setJobId(jobId).build();
        }

        static JobInfo fromPb(Job jobPd)
        {
            return new BuilderImpl(jobPd).build();
        }
    }
}
