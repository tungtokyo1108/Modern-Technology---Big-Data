/**
 * Google - Big Data Technology
 * https://github.com/googleapis/google-cloud-java/blob/master/google-cloud-clients/google-cloud-bigquery/src/main/java/com/google/cloud/bigquery/DatasetInfo.java
 *
 *  Created on: Feb 13, 2019
 *  Student (MIG Virtual Developer): Tung Dang
 */

package com.google.cloud.bigquery.mirror;

import static com.google.common.base.MoreObjects.firstNonNull;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.api.client.util.Data;
import com.google.api.services.bigquery.model.Dataset;
import com.google.api.services.bigquery.model.TableReference;
import com.google.common.base.Function;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.protobuf.DescriptorProtos.FieldDescriptorProto.Label;
import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import javax.xml.crypto.dsig.keyinfo.RetrievalMethod;

public class DatasetInfo implements Serializable 
{
    static final Function<Dataset, DatasetInfo> FROM_PB_FUNCTION = 
        new Function<Dataset,DatasetInfo>() {
            @Override
            public DatasetInfo apply(Dataset pb) {
                return DatasetInfo.fromPb(pb);
            }
        };
    static final Function<DatasetInfo, Dataset> TO_PB_FUNCTION = 
        new Function<DatasetInfo,Dataset>() {
            @Override
            public Dataset apply(DatasetInfo datasetInfo) {
                return datasetInfo.toPb();
            }
        };
    private static final long serialVersionUID = 8469473744160758489L;

    private final DatasetId datasetId;
    private final List<Acl> acl;
    private final Long creationTime;
    private final Long defaultTableLifetime;
    private final String description;
    private final String etag;
    private final String friendlyName;
    private final String generatedId;
    private final Long lastModified;
    private final String location;
    private final String selfLink;
    private final Labels labels;

    public abstract static class Builder 
    {
       public abstract Builder setDatasetId(DatasetId datasetId);
       public abstract Builder setAcl(List<Acl> acl);
       abstract Builder setCreationTime(Long creationTime);
       public abstract Builder setDefaultTableLifetime(Long defaultTableLifetime);
       public abstract Builder setDescription(String description);
       abstract Builder setEtag(String etag);
       public abstract Builder setFriendlyName(String friendlyName);
       abstract Builder setGeneratedId(String generatedId);
       abstract Builder setLastModifier(Long lastModified);
       public abstract Builder setLocation(String location);
       abstract Builder setSelfLink(String selfLink);
       public abstract Builder setLabels(Map<String, String> labels);
       public abstract DatasetInfo build();
    }

    static final class BuilderImpl extends Builder {

        private DatasetId datasetId;
        private List<Acl> acl;
        private Long creationTime;
        private Long defaultTableLifetime;
        private String description;
        private String etag;
        private String friendlyName;
        private String generatedId;
        private Long lastModified;
        private String location;
        private String selfLink;
        private Labels labels = Labels.ZERO; 

        BuilderImpl() {}

        BuilderImpl(DatasetInfo datasetInfo) 
        {
            this.datasetId = datasetInfo.datasetId;
            this.acl = datasetInfo.acl;
            this.creationTime = datasetInfo.creationTime;
            this.defaultTableLifetime = datasetInfo.defaultTableLifetime;
            this.description = datasetInfo.description;
            this.etag = datasetInfo.etag;
            this.friendlyName = datasetInfo.friendlyName;
            this.generatedId = datasetInfo.generatedId;
            this.lastModified = datasetInfo.lastModified;
            this.location = datasetInfo.location;
            this.selfLink = datasetInfo.selfLink;
            this.labels = datasetInfo.labels;
        }

        BuilderImpl(com.google.api.services.bigquery.model.Dataset datasetPb)
        {
            if (datasetPb.getDatasetReference() != null)
            {
                this.datasetId = DatasetId.fromPb(datasetPb.getDatasetReference());
            }
            if (datasetPb.getAccess() != null)
            {
                this.acl = 
                    Lists.transform(
                        datasetPb.getAccess(),
                        new Function<Dataset.Access, Acl>() {
                            @Override
                            public Acl apply(Dataset.Access accessPb) {
                                return Acl.fromPb(accessPb);
                            }
                        }
                    );
            }
            this.creationTime = datasetPb.getCreationTime();
            this.defaultTableLifetime = datasetPb.getDefaultPartitionExpirationMs();
            this.description = datasetPb.getDescription();
            this.etag = datasetPb.getEtag();
            this.friendlyName = datasetPb.getFriendlyName();
            this.generatedId = datasetPb.getId();
            this.lastModified = datasetPb.getLastModifiedTime();
            this.location = datasetPb.getLocation();
            this.selfLink = datasetPb.getSelfLink();
            this.labels = Labels.fromPb(datasetPb.getLabels());
        }

        @Override 
        public Builder setDatasetId(DatasetId datasetId) 
        {
            this.datasetId = checkNotNull(datasetId);
            return this;
        }

        @Override
        public Builder setAcl(List<Acl> acl)
        {
            this.acl = acl != null ? ImmutableList.copyOf(acl) : null;
            return this;
        }

        @Override
        Builder setCreationTime(Long creationTime)
        {
            this.creationTime = creationTime;
            return this;
        }

        @Override
        public Builder setDescription(String description)
        {
            this.description = firstNonNull(description, Data.<String>nullOf(String.class));
            return this;
        }

        @Override
        public Builder setDefaultTableLifetime(Long defaultTableLifetime)
        {
            this.defaultTableLifetime = firstNonNull(defaultTableLifetime, Data.<Long>nullOf(Long.class));
            return this;
        }

        @Override
        Builder setEtag(String etag)
        {
            this.etag = etag;
            return this;
        }

        @Override
        public Builder setFriendlyName(String friendlyName)
        {
            this.friendlyName = firstNonNull(friendlyName, Data.<String>nullOf(String.class));
            return this;
        }

        @Override
        Builder setGeneratedId(String generatedId)
        {
            this.generatedId = generatedId;
            return this;
        }

        @Override 
        Builder setLastModifier(Long lastModifier)
        {
            this.lastModified = lastModifier;
            return this;
        }

        @Override
        public Builder setLocation(String location)
        {
            this.location = firstNonNull(location, Data.<String>nullOf(String.class));
            return this;
        }

        @Override
        Builder setSelfLink(String selfLink)
        {
            this.selfLink = selfLink;
            return this;
        }

        @Override
        public Builder setLabels(Map<String, String> labels)
        {
            this.labels = Labels.fromUser(labels);
            return this;
        }

        @Override
        public DatasetInfo build()
        {
            return new DatasetInfo(this);
        }

        DatasetInfo(BuilderImpl builder) 
        {
            datasetId = builder.datasetId;
            acl = builder.acl;
            creationTime = builder.creationTime;
            defaultTableLifetime = builder.defaultTableLifetime;
            description = builder.description;
            etag = builder.etag;
            friendlyName = builder.friendlyName;
            generatedId = builder.generatedId;
            lastModified = builder.lastModified;
            location = builder.location;
            selfLink = builder.selfLink;
            labels = builder.labels;
        }

        public DatasetId getDatasetId()
        {
            return datasetId;
        }

        public List<Acl> getAcl() 
        {
            return acl;
        }

        public Long getCreationTime()
        {
            return creationTime;
        }

        public Long getDefaultTableLeftime()
        {
            return defaultTableLifetime;
        }

        public String getDescription() 
        {
            return description;
        }

        public String getEtag()
        {
            return etag;
        }

        public String getFriendlyName()
        {
            return friendlyName;
        }

        public String getGeneratedId()
        {
            return generatedId;
        }

        public Long getLastModified() 
        {
            return lastModified;
        }

        public String getLocation() 
        {
            return location;
        }

        static String getSelfLink() 
        {
            return selfLink;
        }

        public Map<String, String> getLabels()
        {
            return labels.userMap();
        }

        public Builder toBuilder()
        {
            return new BuilderImpl(this);
        }

        @Override 
        public String toString()
        {
            return MoreObjects.toStringHelper(this)
                .add("datasetId", datasetId)
                .add("creationTime", creationTime)
                .add("defaultTableLifetime", defaultTableLifetime)
                .add("description", description)
                .add("etag", etag)
                .add("friendltName", friendlyName)
                .add("generatedId", generatedId)
                .add("lastModified", lastModified)
                .add("location", location)
                .add("selfLink", selfLink)
                .add("acl", acl)
                .add("labels", labels)
                .toString();
        }

        @Override
        public int hashCode()
        {
            return Objects.hash(datasetId);
        }

        @Override
        public boolean equals(Object obj)
        {
            return obj == this 
                || obj != null
                    && obj.getClass().equals(DatasetInfo.class)
                    && Objects.equals(toPb(), ((DatasetInfo) obj).toPb());
        }

        DatasetInfo setProjectId(String projectId)
        {
            Builder builder = toBuilder();
            builder.setDatasetId(getDatasetId().setProjectId(projectId));
            if (getAcl() != null)
            {
                List<Acl> acls = Lists.newArrayListWithCapacity(getAcl().size());
                for (Acl acl : getAcl())
                {
                    if (acl.getEntity().getType() == Acl.Entity.Type.VIEW)
                    {
                        Dataset.Access accessPb = acl.toPb();
                        TableReference viewReferencePb = accessPb.getView();
                        if (viewReferencePb.getProjectId() == null)
                        {
                            viewReferencePb.setProjectId(projectId);
                        }
                        acls.add(Acl.of(new Acl.View(TableId.fromPb(viewReferencePb))));
                    } 
                    else
                    {
                        acls.add(acl);
                    } 
                }
                builder.setAcl(acls);
            }
            return builder.build();
        }

        Dataset toPb() 
        {
            Dataset datasetPb = new Dataset();
            datasetPb.setDatasetReference(datasetId.toPb());
            datasetPb.setCreationTime(creationTime);
            datasetPb.setDefaultTableExpirationMs(defaultTableLifetime);
            datasetPb.setDescription(description);
            datasetPb.setEtag(etag);
            datasetPb.setFriendlyName(friendlyName);
            datasetPb.setId(generatedId);
            datasetPb.setLastModifiedTime(lastModified);
            datasetPb.setLocation(location);
            datasetPb.setSelfLink(selfLink);
            if (acl != null)
            {
                datasetPb.setAccess(
                    Lists.transform(
                        acl,
                        new Function<Acl, Dataset.Access>() {
                            @Override
                            public Dataset.Access apply(Acl acl)
                            {
                                return acl.toPb();
                            }
                        }
                    )
                );
            }
            datasetPb.setLabels(labels.toPb());
            return datasetPb;
        }

        public static Builder newBuilder(DatasetId datasetId)
        {
            return new BuilderImpl().setDatasetId(datasetId);
        }

        public static Builder newBuilder(String datasetId)
        {
            return newBuilder(DatasetId.of(datasetId));
        }

        public static Builder newBuilder(String projectId, String datasetId)
        {
            return newBuilder(projectId, datasetId);
        }

        public static DatasetInfo of(DatasetId datasetId)
        {
            return newBuilder(datasetId).build();
        }

        public static DatasetInfo of(String datasetId)
        {
            return newBuilder(datasetId).build();
        }

        static DatasetInfo fromPb(Dataset datasetPb)
        {
            return new BuilderImpl(datasetPb).build();
        }
    }
}
