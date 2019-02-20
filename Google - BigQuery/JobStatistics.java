/**
 * Google - Big Data Technology
 * https://github.com/googleapis/google-cloud-java/blob/master/google-cloud-clients/google-cloud-bigquery/src/main/java/com/google/cloud/bigquery/JobStatistics.java
 *
 *  Created on: Feb 17, 2019
 *  Student (MIG Virtual Developer): Tung Dang
 */

package com.google.cloud.bigquery.mirror;

import com.google.api.core.ApiFunction;
import com.google.api.services.bigquery.model.JobConfiguration;
import com.google.api.services.bigquery.model.JobStatistics2;
import com.google.api.services.bigquery.model.JobStatistics3;
import com.google.api.services.bigquery.model.JobStatistics4;
import com.google.api.services.bigquery.model.Table;
import com.google.cloud.StringEnumType;
import com.google.cloud.StringEnumValue;
import com.google.cloud.bigquery.QueryStage;
import com.google.cloud.bigquery.Schema;
import com.google.cloud.bigquery.TimelineSample;
import com.google.cloud.bigquery.JobStatistics.QueryStatistics.StatementType;
import com.google.common.base.MoreObjects;
import com.google.common.base.MoreObjects.ToStringHelper;
import com.google.common.collect.Lists;
import com.google.errorprone.annotations.OverridingMethodsMustInvokeSuper;
import com.google.rpc.RetryInfo;

import java.beans.Statement;
import java.io.Serializable;
import java.math.BigInteger;
import java.util.List;
import java.util.Objects;

public abstract class JobStatistics implements Serializable 
{
    private static final long serialVersionUID = 1433024714741660399L;

    private final Long creationTime;
    private final Long endTime;
    private final Long startTime;

    public static class CopyStatistics extends JobStatistics
    {
        private static final long serialVersionUID = 8218325588441660938L;

        static final class Builder extends JobStatistics.Builder<CopyStatistics, Builder>
        {
            private Builder() {}
            private Builder(com.google.api.services.bigquery.model.JobStatistics statisticsPb)
            {
                super(statisticsPb);
            }
            @Override
            CopyStatistics build()
            {
                return new CopyStatistics(this);
            }
        }

        private CopyStatistics(Builder builder)
        {
            super(builder);
        }

        @Override
        public final boolean equals(Object obj) {
            return obj == this
                || obj != null
                    && obj.getClass().equals(CopyStatistics.class)
                    && baseEquals((CopyStatistics) obj);
        }

        @Override
        public final int hashCode()
        {
            return baseHashCode();
        }

        static Builder newBuilder()
        {
            return new Builder();
        }

        @SuppressWarnings("unchecked")
        static CopyStatistics fromPb(com.google.api.services.bigquery.model.JobStatistics statisticsPb)
        {
            return new Builder(statisticsPb).build();
        }
    }

    public static class ExtractStatistics extends JobStatistics 
    {
        private static final long serialVersionUID = -1566598819212767373L;

        private final List<Long> destinationUriFileCounts;
        static final class Builder extends JobStatistics.Builder<ExtractStatistics, Builder>
        {
            private List<Long> destinationUriFileCounts;
            private Builder() {}
            private Builder(com.google.api.services.bigquery.model.JobStatistics statisticsPb)
            {
                super(statisticsPb);
                if (statisticsPb.getExtract() != null)
                {
                    this.destinationUriFileCounts = statisticsPb.getExtract().getDestinationUriFileCounts();
                }
            }

            Builder setDestinationFileCounts(List<Long> destinationUriFileCounts)
            {
                this.destinationUriFileCounts = destinationUriFileCounts;
                return self();
            }

            @Override
            ExtractStatistics build()
            {
                return new ExtractStatistics(this);
            }
        }

        private ExtractStatistics(Builder builder)
        {
            super(builder);
            this.destinationUriFileCounts = builder.destinationUriFileCounts;
        }

        public List<Long> getDestinationUriFileCounts()
        {
            return destinationUriFileCounts;
        }

        @Override
        ToStringHelper toStringHelper() 
        {
            return super.toStringHelper().add("destinationUriFileCounts", destinationUriFileCounts);
        }

        @Override
        public final boolean equals(Object obj)
        {
            return obj == this
                || obj != null
                    && obj.getClass().equals(ExtractStatistics.class)
                    && baseEquals((ExtractStatistics) obj);
        }

        @Override
        public final int hashCode()
        {
            return Objects.hash(baseHashCode(), destinationUriFileCounts);
        }

        @Override
        com.google.cloud.bigquery.model.JobStatistics toPb()
        {
            com.google.cloud.bigquery.model.JobStatistics statisticsPb = super.toPb();
            return statisticsPb.setExtract(
                new JobStatistics4().setDestinationUriFileCounts(destinationUriFileCounts));
        }

        static Builder newBuilder()
        {
            return new Builder();
        }

        @SuppressWarnings("unchecked")
        static ExtractStatistics fromPb(
            com.google.api.services.bigquery.model.JobStatistics statisticsPb) {
                return new Builder(statisticsPb).build();
        }
    }

    public static class LoadStatistics extends JobStatistics 
    {
        private static final long serialVersionUID = -707369246536309215L;

        private final Long inputBytes;
        private final Long inputFiles;
        private final Long outputBytes;
        private final Long outputRows;
        private final Long badRecords;

        static final class Builder extends JobStatistics.Builder<LoadStatistics, Builder>
        {
            private Long inputBytes;
            private Long inputFiles;
            private Long outputBytes;
            private Long outputRows;
            private Long badRecords;

            private Builder() {}

            private Builder(com.google.api.services.bigquery.model.JobStatistics statisticsPb)
            {
                super(statisticsPb);
                if (statisticsPb.getLoad() != null)
                {
                    this.inputBytes = statisticsPb.getLoad().getInputFileBytes();
                    this.inputFiles = statisticsPb.getLoad().getInputFiles();
                    this.outputBytes = statisticsPb.getLoad().getOutputBytes();
                    this.outputRows = statisticsPb.getLoad().getOutputRows();
                    this.badRecords = statisticsPb.getLoad().getBadRecords();
                }
            }

            Builder setInputBytes(Long inputBytes)
            {
                this.inputBytes = inputBytes;
                return self();
            }

            Builder setInputFiles(Long inputFiles)
            {
                this.inputFiles = inputFiles;
                return self();
            }

            Builder setOutputBytes(Long outputBytes)
            {
                this.outputBytes = outputBytes;
                return self();
            }

            Builder setOutputRows(Long outputRows)
            {
                this.outputRows = outputRows;
                return self();
            }

            Builder setBadRecords(Long badRecords)
            {
                this.badRecords = badRecords;
                return self();
            }

            @Override
            LoadStatistics build()
            {
                return new LoadStatistics(this);
            }
        }

        private LoadStatistics(Builder builder) 
        {
            super(builder);
            this.inputBytes = builder.inputBytes;
            this.inputFiles = builder.inputFiles;
            this.outputBytes = builder.outputBytes;
            this.outputRows = builder.outputRows;
            this.badRecords = builder.badRecords;
        }

        public Long getInputBytes()
        {
            return inputBytes;
        }

        public Long getInputFiles()
        {
            return inputFiles;
        }

        public Long getOutputBytes()
        {
            return outputBytes;
        }

        public Long getOutputRows()
        {
            return outputRows;
        }

        public Long getBadRecords()
        {
            return badRecords;
        }

        @Override 
        ToStringHelper toStringHelper()
        {
            return super.toStringHelper()
                .add("inputBytes", inputBytes)
                .add("intputFiles", inputFiles)
                .add("outputBytes", outputBytes)
                .add("outputRows", outputRows)
                .add("badRecords", badRecords);
        }

        @Override
        public final boolean equals(Object obj)
        {
            return obj == this
                || obj != null
                    && obj.getClass().equals(LoadStatistics.class)
                    && baseEquals((LoadStatistics) obj);
        }

        @Override
        public final int hashCode()
        {
            return Objects.hash(
                baseHashCode(), inputBytes, inputFiles, outputBytes, outputRows, badRecords
            );
        }

        @Override
        com.google.api.services.bigquery.model.JobStatistics toPb()
        {
            JobStatistics3 loadStatisticsPb = new JobStatistics3();
            loadStatisticsPb.setInputFileBytes(inputBytes);
            loadStatisticsPb.setInputFiles(inputFiles);
            loadStatisticsPb.setOutputBytes(outputBytes);
            loadStatisticsPb.setOutputRows(outputRows);
            loadStatisticsPb.setBadRecords(badRecords);
            return super.toPb().setLoad(loadStatisticsPb);
        }

        static Builder newBuilder() 
        {
            return new Builder();
        }

        @SuppressWarnings("unchecked")
        static LoadStatistics fromPb(com.google.api.services.bigquery.model.JobStatistics statisticsPb)
        {
            return new Builder(statisticsPb).build();
        }
    }

    public static class QueryStatistics extends JobStatistics
    {
        private static final long serialVersionUID = 7539354109226732353L;

        private final Integer billingTier;
        private final Boolean cacheHit;
        private final String ddlOperationPerformed;
        private final TableId ddlTargetTable;
        private final Long estimatedBytesProcessed;
        private final Long numDmlAffectedRows;
        private final List<TableId> referencedTables;
        private final StatementType statementType;
        private final Long totalBytesBilled;
        private final Long totalBytesProcessed;
        private final Long totalPartitionsProcessed;
        private final Long totalSlotMs;
        private final List<QueryStage> queryPlan;
        private final List<TimelineSample> timeLine;
        private final Schema schema;

        public static final class StatementType extends StringEnumType
        {
            private static final long serialVersionUID = 818920627219751204L;

            private static final ApiFunction<String, StatementType> CONSTRUCTOR = 
                new ApiFunction<String, StatementType>() {
                    @Override
                    public StatementType apply(String constant)
                    {
                        return new StatementType(constant);
                    }
                };

            private static final StringEnumType<StatementType> type = 
                new StringEnumType(StatementType.class, CONSTRUCTOR);

            public static final StatementType SELECT = type.createAndRegister("SELECT");
            public static final StatementType UPDATE = type.createAndRegister("UPDATE");
            public static final StatementType INSERT = type.createAndRegister("INSERT");
            public static final StatementType DELETE = type.createAndRegister("DELETE");
            public static final StatementType CREATE_TABLE = type.createAndRegister("CREATE_TABLE");
            public static final StatementType CREATE_TABLE_AS_SELECT = 
                type.createAndRegister("CREATE_TABLE_AS_SELECT");
            public static final StatementType CREATE_VIEW = type.createAndRegister("CREATE_VIEW");
            public static final StatementType DROP_TABLE = type.createAndRegister("DROP_TABLE");
            public static final StatementType DROP_VIEW = type.createAndRegister("DROP_VIEW");
            public static final StatementType MERGE = type.createAndRegister("MERGE");

            private StatementType(String constant)
            {
                super(constant);
            }

            public static StatementType valueOfStrict(String constant)
            {
                return type.valueOfStrict(constant);
            }

            public static StatementType valueOf(String constant)
            {
                return type.valueOf(constant);
            }

            public static StatementType[] values()
            {
                return type.values();
            }
        }

        static final class Builder extends JobStatistics.Builder<QueryStatistics, Builder>
        {
            private Integer billingTier;
            private Boolean cacheHit;
            private String ddlOperationPerformed;
            private TableId ddlTargetTable;
            private Long estimatedBytesProcessed;
            private Long numDmlAffectedRows;
            private List<TableId> referencedTables;
            private StatementType statementType;
            private Long totalBytesBilled;
            private Long totalBytesProcessed;
            private Long totalPartitionsProcessed;
            private Long totalSlotMs;
            private List<QueryStage> queryPlan;
            private List<TimelineSample> timeline;
            private Schema schema;

            private Builder() {}

            private Builder (com.google.api.services.bigquery.model.JobStatistics statisticsPb) 
            {
                super(statisticsPb);
                if (statisticsPb.getQuery() != null)
                {
                    this.billingTier = statisticsPb.getQuery().getBillingTier();
                    this.cacheHit = statisticsPb.getQuery().getCacheHit();
                    this.ddlOperationPerformed = statisticsPb.getQuery().getDdlOperationPerformed();
                    if (statisticsPb.getQuery().getDdlTargetTable() != null)
                    {
                        this.ddlTargetTable = TableId.fromPb(statisticsPb.getQuery().getDdlTargetTable());
                    }
                    this.estimatedBytesProcessed = statisticsPb.getQuery().getEstimatedBytesProcessed();
                    this.numDmlAffectedRows = statisticsPb.getQuery().getNumDmlAffectedRows();
                    this.totalBytesBilled = statisticsPb.getQuery().getTotalBytesBilled();
                    this.totalBytesProcessed = statisticsPb.getQuery().getTotalBytesProcessed();
                    this.totalPartitionsProcessed = statisticsPb.getQuery().getTotalPartitionsProcessed();
                    this.totalSlotMs = statisticsPb.getQuery().getTotalSlotMs();
                    if (statisticsPb.getQuery().getStatementType() != null)
                    {
                        this.statementType = StatementType.valueOf(statisticsPb.getQuery().getStatementType());
                    }
                    if (statisticsPb.getQuery().getReferencedTables() != null)
                    {
                        this.referencedTables = 
                            Lists.transform(statisticsPb.getQuery().getReferencedTables(), TableId.FROM_PB_FUNCTION);
                    }
                    if (statisticsPb.getQuery().getQueryPlan() != null)
                    {
                        this.queryPlan = 
                            Lists.transform(statisticsPb.getQuery().getQueryPlan(), QueryStage.FROM_PB_FUNCTION);
                    }
                    if (statisticsPb.getQuery().getTimeline() != null)
                    {
                        this.timeline = 
                            Lists.transform(statisticsPb.getQuery().getTimeline(), TimelineSample.FROM_PB_FUNCTION);
                    }
                    if (statisticsPb.getQuery().getSchema() != null)
                    {
                        this.schema = Schema.fromPb(statisticsPb.getQuery().getSchema());
                    }
                }
            }

            Builder setBillingTier(Integer billingTier)
            {
                this.billingTier = billingTier;
                return self();
            }

            Builder setCacheHit(Boolean cacheHit)
            {
                this.cacheHit = cacheHit;
                return self();
            }

            Builder setDDLOperationPerformed(String ddlOperationPerformed)
            {
                this.ddlOperationPerformed = ddlOperationPerformed;
                return self();
            }

            Builder setDDLTargetTable(TableId ddlTargetTable)
            {
                this.ddlTargetTable = ddlTargetTable;
                return self();
            }

            Builder setEstimatedBytesProcessed(Long estimatedBytesProcessed)
            {
                this.estimatedBytesProcessed = estimatedBytesProcessed;
                return self();
            }

            Builder setNumDmlAffectedRows(Long numDmlAffectedRows)
            {
                this.numDmlAffectedRows = numDmlAffectedRows;
                return self();
            }

            Builder setReferenceTables(List<TableId> referenceTables)
            {
                this.referencedTables = referenceTables;
                return self();
            }

            Builder setStatementType(StatementType statementType)
            {
                this.statementType = statementType;
                return self();
            } 

            Builder setStatementType(String strStatementType)
            {
                this.statementType = StatementType.valueOf(strStatementType);
                return self();
            }

            Builder setTotalBytesBilled(Long totalBytesBilled)
            {
                this.totalBytesBilled = totalBytesBilled;
                return self();
            }

            Builder setTotalBytesProcessed(Long totalBytesProcessed)
            {
                this.totalBytesProcessed = totalBytesProcessed;
                return self();
            }

            Builder setTotalPartitionsProcessed(Long totalPartitionProcessed)
            {
                this.totalPartitionsProcessed = totalPartitionProcessed;
                return self();
            }

            Builder setTotalSlotMs(Long totalSlotMs)
            {
                this.totalSlotMs = totalSlotMs;
                return self();
            }

            Builder setQueryPlan(List<QueryStage> queryPlan)
            {
                this.queryPlan = queryPlan;
                return self();
            }

            Builder setTimeline(List<TimelineSample> timeline)
            {
                this.timeline = timeline;
                return self();
            }

            Builder setSchema(Schema schema)
            {
                this.schema = schema;
                return self();
            }

            @Override
            QueryStatistics build() 
            {
                return new QueryStatistics(this);
            }
        }

        private QueryStatistics(Builder builder)
        {
            super(builder);
            this.billingTier = builder.billingTier;
            this.cacheHit = builder.cacheHit;
            this.ddlOperationPerformed = builder.ddlOperationPerformed;
            this.ddlTargetTable = builder.ddlTargetTable;
            this.estimatedBytesProcessed = builder.estimatedBytesProcessed;
            this.numDmlAffectedRows = builder.numDmlAffectedRows;
            this.referencedTables = builder.referencedTables;
            this.statementType = builder.statementType;
            this.totalBytesBilled = builder.totalBytesBilled;
            this.totalBytesProcessed = builder.totalBytesProcessed;
            this.totalPartitionsProcessed = builder.totalPartitionsProcessed;
            this.totalSlotMs = builder.totalSlotMs;
            this.queryPlan = builder.queryPlan;
            this.timeLine = builder.timeline;
            this.schema = builder.schema;
        }

        public Integer getBillingTier()
        {
            return billingTier;
        }

        public Boolean getCacheHit()
        {
            return cacheHit;
        }

        public String getDdlOperationPerformed()
        {
            return ddlOperationPerformed;
        }

        public TableId getDdlTargetTable()
        {
            return ddlTargetTable;
        }

        public Long getEstimatedBytesProcessed() 
        {
            return estimatedBytesProcessed;
        }

        public Long getNumDmlAffectedRows()
        {
            return numDmlAffectedRows;
        }

        public List<TableId> getReferencedTables()
        {
            return referencedTables;
        }

        public StatementType getStatementType()
        {
            return statementType;
        }

        public Long getTotalBytesProcessed()
        {
            return totalBytesProcessed;
        }

        public Long getTotalPartitionsProcessed()
        {
            return totalPartitionsProcessed;
        }

        public Long getTotalSlotMs()
        {
            return totalSlotMs;
        }

        public List<QueryStage> getQueryPlan()
        {
            return queryPlan;
        }

        public List<TimelineSample> getTimeline()
        {
            return timeLine;
        }

        public Schema getSchema() 
        {
            return schema;
        }

        @Override
        ToStringHelper toStringHelper()
        {
            return super.toStringHelper()
                .add("billingTier", billingTier)
                .add("cacheHit", cacheHit)
                .add("totalBytesBilled", totalBytesBilled)
                .add("totalBytesProcessed", totalBytesProcessed)
                .add("queryPlan", queryPlan)
                .add("timeline", timeLine)
                .add("schema", schema);
        }

        @Override
        public final boolean equals(Object obj)
        {
            return obj == this
                || obj != null
                && obj.getClass().equals(QueryStatistics.class)
                && baseEquals((QueryStatistics) obj);
        }

        @Override 
        public final int hashCode()
        {
            return Objects.hash(
                baseHashCode(),
                billingTier,
                cacheHit,
                totalBytesBilled,
                totalBytesProcessed,
                queryPlan,
                schema
            );
        }

        
    }
}
