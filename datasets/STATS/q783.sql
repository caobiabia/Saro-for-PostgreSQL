select  count(*) from comments as c,  		posts as p,          postHistory as ph,          votes as v,          badges as b,          users as u  where u.Id = p.OwnerUserId     and u.Id = b.UserId     and p.Id = c.PostId     and p.Id = ph.PostId     and p.Id = v.PostId  AND b.Date>='2010-07-22 12:38:19'::timestamp  AND b.Date<='2014-09-10 17:09:59'::timestamp  AND c.Score=0  AND c.CreationDate<='2014-09-12 11:18:24'::timestamp  AND p.PostTypeId=2  AND p.Score>=-3  AND p.Score<=17  AND p.AnswerCount<=4  AND p.CommentCount>=0  AND p.CommentCount<=17  AND u.Reputation>=1  AND u.DownVotes>=0  AND u.UpVotes>=0;