select  count(*) from postHistory as ph,  		posts as p,          users as u where ph.PostId = p.Id 	and p.OwnerUserId = u.Id  AND ph.CreationDate>='2010-10-20 08:22:36'::timestamp  AND ph.CreationDate<='2014-08-15 16:03:37'::timestamp  AND p.Score>=-7  AND u.Views<=74  AND u.DownVotes>=0;