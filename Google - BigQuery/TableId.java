/**
 * Google - Big Data Technology
 * https://github.com/googleapis/google-cloud-java/blob/master/google-cloud-clients/google-cloud-bigquery/src/main/java/com/google/cloud/bigquery/TableId.java
 *
 *  Created on: Jan 13, 2019
 *  Student (MIG Virtual Developer): Tung Dang
 */

package com.google.cloud.bigquery.mirror;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.api.services.bigquery.model.TableReference;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;

import java.io.Serializable;
import java.util.Objects;

public final class TableId implements Serializable 
{
    static final Function<TableReference, TableId> FROM_PB_FUNCTION = 
        new Function<TableReference,TableId>() {
            @Override
            public TableId apply(TableReference pb) 
            {
                return TableId.fromPb(pb);
            }
        };    

    static final Function<TableId, TableReference> TO_PB_FUNCTION = 
        new Function<TableId, TableReference>() {
            @Override
            public TableReference apply(TableId tableId)
            {
                return tableId.toPb();
            }
        };

    private static final long serialVersionUID = -6186254820908152300L;

    private final String project;
    private final String dataset;
    private final String table;

    public String getProject() 
    {
        return dataset;
    }

    public String getDataset()
    {
        return dataset;
    }

    public String getTable()
    {
        return table;
    }

    public TableId(String project, String dataset, String table) 
    {
        this.project = project;
        this.dataset = dataset;
        this.table = table;
    }

    public static TableId of(String project, String dataset, String table)
    {
        return new TableId(checkNotNull(project), checkNotNull(dataset), checkNotNull(table));
    }

    public static TableId of(String dataset, String table)
    {
        return new TableId(null, checkNotNull(dataset), checkNotNull(table));
    }

    @Override
    public int hashCode()
    {
        return Objects.hash(project, dataset, table);
    }

    @Override
    public String toString() 
    {
        return toPb().toString();
    }

    TableId setProjectId(String projectId)
    {
        Preconditions.checkArgument(!Strings.isNullOrEmpty(projectId), "Provided projectId is null or empty");
        return TableId.of(projectId, getDataset(), getTable());
    }

    TableReference toPb()
    {
        return new TableReference().setProjectId(project).setDatasetId(dataset).setTableId(table);
    }

    static TableId fromPb(TableReference tableRef)
    {
        return new TableId(tableRef.getProjectId(), tableRef.getDatasetId(), tableRef.getTableId());
    }
}